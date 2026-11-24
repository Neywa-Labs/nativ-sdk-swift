import Foundation
import MLX
import MLXNN

public struct NemotronLabsDiffusionConfiguration: Decodable, Sendable {
    public let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let hiddenActivation: String
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let padTokenID: Int?
    public let bosTokenID: Int?
    public let eosTokenID: IntOrIntArray
    public let tieWordEmbeddings: Bool
    public let ropeTheta: Float
    public let ropeParameters: [String: StringOrNumber]?
    public let attentionBias: Bool
    public let mlpBias: Bool
    public let maskTokenID: Int
    public let defaultGenerationMode: String
    public let defaultDiffusionSampler: String
    public let defaultDiffusionSteps: Int
    public let defaultDiffusionThreshold: Float?
    public let defaultDiffusionMinThreshold: Float?
    public let defaultDiffusionSamplingScalingFactor: Float
    public let dlmParadigm: String
    public let blockSize: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenActivation = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case padTokenID = "pad_token_id"
        case bosTokenID = "bos_token_id"
        case eosTokenID = "eos_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case maskTokenID = "mask_token_id"
        case defaultGenerationMode = "default_generation_mode"
        case defaultDiffusionSampler = "default_diffusion_sampler"
        case defaultDiffusionSteps = "default_diffusion_steps"
        case defaultDiffusionThreshold = "default_diffusion_threshold"
        case defaultDiffusionMinThreshold = "default_diffusion_min_threshold"
        case defaultDiffusionSamplingScalingFactor = "default_diffusion_sampling_scaling_factor"
        case dlmParadigm = "dlm_paradigm"
        case blockSize = "block_size"
    }

    public init(
        modelType: String = "nemotron_labs_diffusion",
        vocabularySize: Int = 131_072,
        hiddenSize: Int = 4096,
        intermediateSize: Int = 14336,
        hiddenLayers: Int = 34,
        attentionHeads: Int = 32,
        kvHeads: Int = 8,
        headDim: Int? = nil,
        hiddenActivation: String = "silu",
        maxPositionEmbeddings: Int = 262_144,
        rmsNormEps: Float = 1e-5,
        padTokenID: Int? = nil,
        bosTokenID: Int? = 1,
        eosTokenID: IntOrIntArray? = nil,
        tieWordEmbeddings: Bool = false,
        ropeTheta: Float = 1_000_000,
        ropeParameters: [String: StringOrNumber]? = nil,
        attentionBias: Bool = false,
        mlpBias: Bool = false,
        maskTokenID: Int = 100,
        defaultGenerationMode: String = "ar",
        defaultDiffusionSampler: String = "native",
        defaultDiffusionSteps: Int = 32,
        defaultDiffusionThreshold: Float? = 0.9,
        defaultDiffusionMinThreshold: Float? = 0.45,
        defaultDiffusionSamplingScalingFactor: Float = 2.0,
        dlmParadigm: String = "bidirectional",
        blockSize: Int = 32,
    ) {
        self.modelType = modelType
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.headDim = headDim ?? hiddenSize / attentionHeads
        self.hiddenActivation = hiddenActivation
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.padTokenID = padTokenID
        self.bosTokenID = bosTokenID
        self.eosTokenID = eosTokenID ?? IntOrIntArray(11)
        self.tieWordEmbeddings = tieWordEmbeddings
        self.ropeTheta = ropeTheta
        self.ropeParameters = ropeParameters
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
        self.maskTokenID = maskTokenID
        self.defaultGenerationMode = defaultGenerationMode
        self.defaultDiffusionSampler = defaultDiffusionSampler
        self.defaultDiffusionSteps = defaultDiffusionSteps
        self.defaultDiffusionThreshold = defaultDiffusionThreshold
        self.defaultDiffusionMinThreshold = defaultDiffusionMinThreshold
        self.defaultDiffusionSamplingScalingFactor = defaultDiffusionSamplingScalingFactor
        self.dlmParadigm = dlmParadigm
        self.blockSize = blockSize
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        let attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        let rawRopeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        var ropeParameters =
            try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
            ?? c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        if ropeParameters == nil {
            ropeParameters = [
                "rope_type": .string("default"),
                "rope_theta": .float(rawRopeTheta),
            ]
        }
        let resolvedRopeTheta = ropeParameters?["rope_theta"]?.asFloat() ?? rawRopeTheta

        try self.init(
            modelType:
                c.decodeIfPresent(String.self, forKey: .modelType)
                ?? "nemotron_labs_diffusion",
            vocabularySize: c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 131_072,
            hiddenSize: hiddenSize,
            intermediateSize:
                c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14336,
            hiddenLayers: c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 34,
            attentionHeads: attentionHeads,
            kvHeads: c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8,
            headDim:
                c.decodeIfPresent(Int.self, forKey: .headDim)
                ?? hiddenSize / attentionHeads,
            hiddenActivation:
                c.decodeIfPresent(String.self, forKey: .hiddenActivation) ?? "silu",
            maxPositionEmbeddings:
                c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262_144,
            rmsNormEps: c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5,
            padTokenID: c.decodeIfPresent(Int.self, forKey: .padTokenID),
            bosTokenID: c.decodeIfPresent(Int.self, forKey: .bosTokenID) ?? 1,
            eosTokenID: c.decodeIfPresent(IntOrIntArray.self, forKey: .eosTokenID),
            tieWordEmbeddings:
                c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false,
            ropeTheta: resolvedRopeTheta,
            ropeParameters: ropeParameters,
            attentionBias: c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false,
            mlpBias: c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false,
            maskTokenID: c.decodeIfPresent(Int.self, forKey: .maskTokenID) ?? 100,
            defaultGenerationMode:
                c.decodeIfPresent(String.self, forKey: .defaultGenerationMode) ?? "ar",
            defaultDiffusionSampler:
                c.decodeIfPresent(String.self, forKey: .defaultDiffusionSampler) ?? "native",
            defaultDiffusionSteps:
                c.decodeIfPresent(Int.self, forKey: .defaultDiffusionSteps) ?? 32,
            defaultDiffusionThreshold:
                c.decodeIfPresent(Float.self, forKey: .defaultDiffusionThreshold),
            defaultDiffusionMinThreshold:
                c.decodeIfPresent(Float.self, forKey: .defaultDiffusionMinThreshold),
            defaultDiffusionSamplingScalingFactor:
                c.decodeIfPresent(
                    Float.self,
                    forKey: .defaultDiffusionSamplingScalingFactor,
                ) ?? 2.0,
            dlmParadigm: c.decodeIfPresent(String.self, forKey: .dlmParadigm)
                ?? "bidirectional",
            blockSize: c.decodeIfPresent(Int.self, forKey: .blockSize) ?? 32,
        )
    }
}

private enum NemotronLabsDiffusionLanguage {
    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        init(_ config: NemotronLabsDiffusionConfiguration) {
            _gateProj.wrappedValue = Linear(
                config.hiddenSize,
                config.intermediateSize,
                bias: config.mlpBias,
            )
            _upProj.wrappedValue = Linear(
                config.hiddenSize,
                config.intermediateSize,
                bias: config.mlpBias,
            )
            _downProj.wrappedValue = Linear(
                config.intermediateSize,
                config.hiddenSize,
                bias: config.mlpBias,
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(silu(gateProj(x)) * upProj(x))
        }
    }

    final class Attention: Module {
        let config: NemotronLabsDiffusionConfiguration
        let scale: Float
        let rope: RoPELayer

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        init(_ config: NemotronLabsDiffusionConfiguration) {
            self.config = config
            scale = pow(Float(config.headDim), -0.5)
            _qProj.wrappedValue = Linear(
                config.hiddenSize,
                config.attentionHeads * config.headDim,
                bias: false,
            )
            _kProj.wrappedValue = Linear(
                config.hiddenSize,
                config.kvHeads * config.headDim,
                bias: false,
            )
            _vProj.wrappedValue = Linear(
                config.hiddenSize,
                config.kvHeads * config.headDim,
                bias: false,
            )
            _oProj.wrappedValue = Linear(
                config.attentionHeads * config.headDim,
                config.hiddenSize,
                bias: config.attentionBias,
            )
            rope = initializeRope(
                dims: config.headDim,
                base: config.ropeTheta,
                traditional: false,
                scalingConfig: config.ropeParameters,
                maxPositionEmbeddings: config.maxPositionEmbeddings,
            )
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            let batchSize = x.dim(0)
            let sequenceLength = x.dim(1)
            var queries = qProj(x).reshaped(
                batchSize,
                sequenceLength,
                config.attentionHeads,
                config.headDim,
            )
            var keys = kProj(x).reshaped(
                batchSize,
                sequenceLength,
                config.kvHeads,
                config.headDim,
            )
            var values = vProj(x).reshaped(
                batchSize,
                sequenceLength,
                config.kvHeads,
                config.headDim,
            )

            queries = queries.transposed(0, 2, 1, 3)
            keys = keys.transposed(0, 2, 1, 3)
            values = values.transposed(0, 2, 1, 3)

            queries = applyRotaryPosition(rope, to: queries, offset: cache?.ropeOffset)
            keys = applyRotaryPosition(rope, to: keys, offset: cache?.ropeOffset)

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, sequenceLength, -1)
            return oProj(output)
        }
    }

    final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: NemotronLabsDiffusionConfiguration) {
            _selfAttention.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(config)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            let h = x + selfAttention(inputLayerNorm(x), mask: mask, cache: cache)
            return h + mlp(postAttentionLayerNorm(h))
        }
    }

    final class Encoder: Module {
        let config: NemotronLabsDiffusionConfiguration

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo var layers: [DecoderLayer]
        @ModuleInfo var norm: RMSNorm

        init(_ config: NemotronLabsDiffusionConfiguration) {
            self.config = config
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
                DecoderLayer(config)
            }
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray,
            cache: [KVCache]? = nil,
            useCausalMask: Bool = true,
        )
            -> MLXArray
        {
            var hidden = embedTokens(inputs)
            let mask =
                useCausalMask
                ? createAttentionMask(h: hidden, cache: cache?.first)
                : MLXFast.ScaledDotProductAttentionMaskMode.none
            for (index, layer) in layers.enumerated() {
                hidden = layer(hidden, mask: mask, cache: cache?[index])
            }
            return norm(hidden)
        }
    }

    final class LanguageModel: Module {
        let config: NemotronLabsDiffusionConfiguration

        @ModuleInfo var model: Encoder
        @ModuleInfo(key: "diffusion_head") var diffusionHead: Linear?

        init(_ config: NemotronLabsDiffusionConfiguration) {
            self.config = config
            _model.wrappedValue = Encoder(config)
            if !config.tieWordEmbeddings {
                _diffusionHead.wrappedValue = Linear(
                    config.hiddenSize,
                    config.vocabularySize,
                    bias: false,
                )
            }
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray,
            cache: [KVCache]? = nil,
            useCausalMask: Bool = true,
        )
            -> MLXArray
        {
            project(model(inputs, cache: cache, useCausalMask: useCausalMask))
        }

        private func project(_ hidden: MLXArray) -> MLXArray {
            if config.tieWordEmbeddings {
                return model.embedTokens.asLinear(hidden)
            }
            return diffusionHead!(hidden)
        }
    }
}

public final class NemotronLabsDiffusion: Module, LLMModel, KVCacheDimensionProvider {
    public var toolCallFormat: ToolCallFormat? { .xmlFunction }

    public let vocabularySize: Int
    public let kvHeads: [Int]
    private var cachedLoraLayers: [LoRALayer]?

    public let config: NemotronLabsDiffusionConfiguration

    @ModuleInfo(key: "language_model")
    private var languageModel: NemotronLabsDiffusionLanguage.LanguageModel

    public init(_ config: NemotronLabsDiffusionConfiguration) {
        precondition(
            config.dlmParadigm == "bidirectional" || config.dlmParadigm == "autoregressive",
            "Unsupported Nemotron Labs Diffusion paradigm: \(config.dlmParadigm)",
        )
        self.config = config
        vocabularySize = config.vocabularySize
        kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        _languageModel.wrappedValue = NemotronLabsDiffusionLanguage.LanguageModel(config)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache, useCausalMask: true)
    }

    /// Reset the cached LoRA-layer list. Call after attaching, fusing, or
    /// removing an adapter so the next toggle re-discovers layers.
    public func invalidateLoRACache() {
        cachedLoraLayers = nil
    }

    public func prepare() throws {
        invalidateLoRACache()
    }

    /// Fast LoRA toggle. Walks `namedModules()` once on first call (or after
    /// `invalidateLoRACache()`) and caches the resulting `[LoRALayer]` list.
    fileprivate func setLoRAEnabledFast(_ enabled: Bool) {
        if cachedLoraLayers == nil {
            var found: [LoRALayer] = []
            for (_, module) in self.namedModules() {
                if let layer = module as? LoRALayer {
                    found.append(layer)
                }
            }
            cachedLoraLayers = found
        }
        // Empty cache means no LoRA layers attached — nothing to do.
        guard let layers = cachedLoraLayers, !layers.isEmpty else { return }
        for layer in layers {
            // `LoRALayer` inherits from `Module` (a class), so the setter
            // mutates the class instance through the protocol existential.
            layer.loraEnabled = enabled
        }
    }

    /// Bidirectional block forward against an optional cached prefix.
    public func bidirectionalLogits(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        languageModel(inputs, cache: cache, useCausalMask: false)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains("rotary_emb.inv_freq") || key.contains("position_ids") {
                continue
            }

            var newKey = key
            if newKey.hasPrefix("encoder.") {
                newKey = "language_model.model.\(newKey.dropFirst("encoder.".count))"
            } else if newKey.hasPrefix("diffusion_head.") {
                newKey = "language_model.\(newKey)"
            } else if newKey.hasPrefix("model.") {
                newKey = "language_model.\(newKey)"
            }

            if config.tieWordEmbeddings, newKey == "language_model.diffusion_head.weight" {
                continue
            }
            sanitized[newKey] = value
        }
        return sanitized
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

extension NemotronLabsDiffusion: MaskedDiffusionLanguageModel {
    public var diffusionMaskTokenID: Int {
        config.maskTokenID
    }

    public var diffusionEOSTokenIDs: [Int] {
        config.eosTokenID.values
    }

    public var diffusionPrefersDiffusionGeneration: Bool {
        config.defaultGenerationMode != "ar"
    }

    public func diffusionLogits(_ tokens: MLXArray) -> MLXArray {
        languageModel(tokens, cache: nil, useCausalMask: false)
    }
}

extension NemotronLabsDiffusion {
    private func makeUnboundedCache() -> [KVCache] {
        kvHeads.map { _ in KVCacheSimple() }
    }

    /// Number of tokens to commit per denoising step within a single block,
    /// distributed evenly with leftovers spread to the first few steps. Mirrors
    /// `_get_num_transfer_tokens` in the Python reference.
    fileprivate static func numTransferTokensPerStep(
        maskCount: Int, steps: Int
    ) -> [Int] {
        precondition(steps > 0)
        let base = maskCount / steps
        let remainder = maskCount % steps
        var schedule = Array(repeating: base, count: steps)
        for i in 0 ..< remainder {
            schedule[i] += 1
        }
        return schedule
    }

    /// Greedy token IDs — argmax over the last axis, no probabilities.
    /// Cheapest option, used by the verify pass and the threshold==0 path
    /// where confidences aren't needed.
    fileprivate static func greedyTokens(logits: MLXArray) -> MLXArray {
        MLX.argMax(logits, axis: -1)
    }

    /// Greedy token IDs plus per-position confidence (probability of the
    /// argmax token).
    ///
    /// Equivalent to `(argmax(logits), softmax(logits).max(-1))` but avoids
    /// materializing the full softmax tensor — for vocab=131072 that's a
    /// ~131072× reduction in float ops per step. Computed via the
    /// log-sum-exp identity: `p_max = exp(max_logit - logSumExp(logits))`.
    fileprivate static func greedyTokensWithConfidence(
        logits: MLXArray
    ) -> (tokens: MLXArray, confidence: MLXArray) {
        let tokens = MLX.argMax(logits, axis: -1)
        let maxLogits = MLX.max(logits, axis: -1)
        let logZ = MLX.logSumExp(logits, axis: -1)
        let confidence = MLX.exp(maxLogits - logZ)
        return (tokens, confidence)
    }
}

// MARK: - AR Generation (model-level helper)

extension NemotronLabsDiffusion {

    /// Pure autoregressive generation (causal attention).
    ///
    /// Mirrors `ar_generate` in `modeling_nemotron_labs_diffusion.py`. Returns
    /// the prompt concatenated with newly generated tokens, and the number of
    /// forward passes performed.
    ///
    /// > Note: This method is provided for **symmetry** with
    /// > ``diffusionGenerate(promptIds:maxNewTokens:blockLength:threshold:eosTokenId:onStep:)``
    /// > and ``linearSpecGenerate(promptIds:maxNewTokens:blockLength:threshold:eosTokenId:onAccept:)``
    /// > so callers can pick a mode by method name. For real applications
    /// > prefer driving AR generation through the standard pipeline
    /// > (`ModelSession` / `NativModel.generate(...)`), which calls
    /// > ``callAsFunction(_:cache:)`` token-by-token via `AutoregressiveIterator` and
    /// > adds streaming, temperature / top-p / repetition-penalty sampling,
    /// > chunked prompt prefill, sliding-window and quantized KV caches, and
    /// > wired-memory ticket coordination — none of which this method does.
    public func arGenerate(
        promptIds: [Int],
        maxNewTokens: Int = 128,
        eosTokenId: Int? = nil
    ) -> (tokens: [Int], nfe: Int) {
        let cache = makeUnboundedCache()

        let prompt = MLXArray(promptIds.map { Int32($0) })[.newAxis, 0...]
        var logits = self(prompt, cache: cache)
        eval(cache, logits)
        var nfe = 1

        var generated: [Int] = []
        for _ in 0 ..< maxNewTokens {
            let lastLogits = logits[0..., -1, 0...]
            let next = Self.greedyTokens(logits: lastLogits)
            let nextInt = Int(next.item(Int32.self))
            generated.append(nextInt)
            if let eos = eosTokenId, nextInt == eos { break }

            let nextArr = MLXArray([Int32(nextInt)])[.newAxis, 0...]
            logits = self(nextArr, cache: cache)
            eval(cache, logits)
            nfe += 1
        }

        return (promptIds + generated, nfe)
    }
}

// MARK: - Diffusion Generation

extension NemotronLabsDiffusion {

    /// Block-wise diffusion (parallel) decoding.
    ///
    /// Mirrors `generate(...)` in `modeling_nemotron_labs_diffusion.py` with
    /// `causal_context = False`. The prompt is causally prefilled into a
    /// shared KV cache once. Each block of `blockLength` tokens is initialized
    /// to `mask_token_id` and refined in up to `blockLength` denoising steps:
    /// at each step the block is forwarded bidirectionally, attending against
    /// the cached prefix without re-running it. After each step the cache is
    /// trimmed by `blockLength` so the next step writes over the same slots.
    /// Once a block stabilizes, we run one causal forward over its finalized
    /// tokens to commit their K/V into the cache for subsequent blocks.
    ///
    public func diffusionGenerate(
        promptIds: [Int],
        maxNewTokens: Int,
        blockLength: Int = 32,
        threshold: Float? = 0.9,
        eosTokenId: Int? = nil,
        onStep: ((_ tokens: [Int], _ nfe: Int) -> Void)? = nil
    ) -> (tokens: [Int], nfe: Int) {
        precondition(
            maxNewTokens % blockLength == 0,
            "maxNewTokens must be a multiple of blockLength")

        let maskId = Int32(config.maskTokenID)
        let numBlocks = maxNewTokens / blockLength

        var xAccum: [Int32] = promptIds.map { Int32($0) }
        var nfe = 0

        // Causal prefill of the prompt once. After this, cache.offset == promptIds.count.
        let cache = makeUnboundedCache()
        let promptArr = MLXArray(promptIds.map { Int32($0) })[.newAxis, 0...]
        _ = self(promptArr, cache: cache)
        eval(cache)
        nfe += 1

        for _ in 0 ..< numBlocks {
            xAccum.append(contentsOf: Array(repeating: maskId, count: blockLength))

            let blockStart = xAccum.count - blockLength
            let stepsPerBlock = blockLength

            let initialMaskCount = (0 ..< blockLength).filter {
                xAccum[blockStart + $0] == maskId
            }.count
            let schedule = Self.numTransferTokensPerStep(
                maskCount: initialMaskCount, steps: stepsPerBlock)

            for step in 0 ..< stepsPerBlock {
                let stillMasked = (0 ..< blockLength).filter {
                    xAccum[blockStart + $0] == maskId
                }
                if stillMasked.isEmpty { break }

                // Forward only the block tokens, bidirectional, against the
                // already-cached prefix. The cache writes the block's K/V at
                // the end — we trim those out below before the next step.
                let blockSlice = xAccum[blockStart ..< (blockStart + blockLength)]
                let blockTensor = MLXArray(Array(blockSlice))[.newAxis, 0...]
                let blockLogits = self.bidirectionalLogits(blockTensor, cache: cache)
                eval(blockLogits)
                nfe += 1

                // For the threshold==0 path we only need a relative ranking
                // among masked positions, so raw max-logits suffice (they're
                // monotonically related to probabilities). For threshold>0 we
                // need actual probability values, computed cheaply via
                // logSumExp instead of materializing the full softmax.
                let useConfidence = (threshold ?? 0) > 0
                let sampledTokens: MLXArray
                let scores: MLXArray
                if useConfidence {
                    let (t, c) = Self.greedyTokensWithConfidence(logits: blockLogits)
                    sampledTokens = t
                    scores = c
                } else {
                    sampledTokens = Self.greedyTokens(logits: blockLogits)
                    scores = MLX.max(blockLogits, axis: -1)
                }
                eval(sampledTokens, scores)

                let scoreArr = scores[0].asArray(Float.self)
                let tokenArr = sampledTokens[0].asArray(Int32.self)
                var candidates: [(pos: Int, score: Float, token: Int32)] = []
                for p in stillMasked {
                    candidates.append((pos: p, score: scoreArr[p], token: tokenArr[p]))
                }

                var commit: [(pos: Int, token: Int32)] = []
                if let thr = threshold, thr > 0 {
                    let thresholded = candidates.filter { $0.score >= thr }
                    if thresholded.isEmpty {
                        if let best = candidates.max(by: { $0.score < $1.score }) {
                            commit.append((pos: best.pos, token: best.token))
                        }
                    } else {
                        for c in thresholded {
                            commit.append((pos: c.pos, token: c.token))
                        }
                    }
                } else {
                    let n = schedule[step]
                    let sorted = candidates.sorted { $0.score > $1.score }
                    for c in sorted.prefix(n) {
                        commit.append((pos: c.pos, token: c.token))
                    }
                }

                for c in commit {
                    xAccum[blockStart + c.pos] = c.token
                }

                // Trim the speculative block K/V from the cache before the
                // next denoising step (or before the post-block causal commit
                // below). This is the cache-reuse trick: we pay one prefix
                // prefill, then each denoising step is O(blockLength) tokens.
                for c in cache { _ = c.trim(blockLength) }

                onStep?(xAccum.map { Int($0) }, nfe)

                if let eos = eosTokenId {
                    let eosI32 = Int32(eos)
                    if let firstEos = (0 ..< blockLength).firstIndex(where: {
                        xAccum[blockStart + $0] == eosI32
                    }) {
                        let beforeEos = (0 ..< firstEos).contains {
                            xAccum[blockStart + $0] == maskId
                        }
                        if !beforeEos { break }
                    }
                }
            }

            // Block has stabilized. Commit its finalized K/V into the cache
            // with a causal forward so subsequent blocks see the right
            // attention pattern from this block.
            let finalizedSlice = xAccum[blockStart ..< (blockStart + blockLength)]
            let finalizedTensor = MLXArray(Array(finalizedSlice))[.newAxis, 0...]
            _ = self(finalizedTensor, cache: cache)
            eval(cache)
            nfe += 1

            if let eos = eosTokenId {
                let eosI32 = Int32(eos)
                let genStart = promptIds.count
                if let firstEos = xAccum[genStart...].firstIndex(of: eosI32) {
                    let cutoff = firstEos + 1
                    return (xAccum[..<cutoff].map { Int($0) }, nfe)
                }
            }
        }

        return (xAccum.map { Int($0) }, nfe)
    }
}

// MARK: - Linear Self-Speculation

extension NemotronLabsDiffusion {

    /// Linear speculative decoding: bidirectional drafter + AR verifier with a
    /// shared KV cache. Mirrors `linear_spec_generate(...)`.
    ///
    /// Requires `batch_size == 1`. The drafter runs with bidirectional
    /// attention (no KV-cache update) over a `blockLength`-token block; the
    /// verifier runs the same block causally and **does** update the cache.
    /// Accepted tokens are the longest matching prefix (plus one bonus AR
    /// token), and the cache is trimmed to length to discard rejected tokens.
    public func linearSpecGenerate(
        promptIds: [Int],
        maxNewTokens: Int = 128,
        blockLength: Int = 32,
        threshold: Float = 0.0,
        eosTokenId: Int? = nil,
        onAccept: ((_ tokens: [Int], _ nfe: Int) -> Void)? = nil
    ) -> (tokens: [Int], nfe: Int) {
        let maskId = Int32(config.maskTokenID)

        let cache = makeUnboundedCache()

        // Causal prefill: LoRA off so the cached prefix matches AR mode.
        setLoRAEnabledFast(false)
        let prompt = MLXArray(promptIds.map { Int32($0) })[.newAxis, 0...]
        var logits = self(prompt, cache: cache)
        eval(cache, logits)
        var nfe = 1

        let lastLogits = logits[0..., -1, 0...]
        let firstNext = Self.greedyTokens(logits: lastLogits)
        var nextToken = Int32(firstNext.item(Int32.self))

        var generated: [Int32] = [nextToken]
        onAccept?(promptIds + generated.map { Int($0) }, nfe)
        if let eos = eosTokenId, Int(nextToken) == eos {
            return (promptIds + [Int(nextToken)], nfe)
        }

        var totalGen = 1

        while totalGen < maxNewTokens {
            let cacheLenBefore = cache[0].offset

            // Build a draft block of size blockLength: position 0 holds the
            // last accepted next-token, the rest are masked.
            var block = [Int32](repeating: maskId, count: blockLength)
            block[0] = nextToken

            // Draft phase — LoRA on, bidirectional, against the causal
            // cache. The encoder sees only the block; the prefix is read from
            // the cache we built during prefill + previous accepted blocks.
            // Each draft forward writes K/V for the speculative block; we
            // trim it back before the next iteration so only verified tokens
            // persist.
            setLoRAEnabledFast(true)
            while true {
                let isMasked = block.contains(maskId)
                if !isMasked { break }

                let blockTensor = MLXArray(block)[.newAxis, 0...]
                let draftLogits = self.bidirectionalLogits(
                    blockTensor, cache: cache)
                eval(draftLogits)
                nfe += 1

                // The threshold==0 path commits everything in one shot and
                // breaks out — no confidence values needed. Only the
                // threshold>0 path pays for the (cheap) per-position prob.
                if threshold > 0 {
                    let (sampledTokens, confidences) =
                        Self.greedyTokensWithConfidence(logits: draftLogits)
                    eval(sampledTokens, confidences)
                    let tokenArr = sampledTokens[0].asArray(Int32.self)
                    let confArr = confidences[0].asArray(Float.self)

                    // Roll the cache back before the next iteration.
                    for c in cache { _ = c.trim(blockLength) }

                    var anyCommitted = false
                    var bestIdx = -1
                    var bestConf: Float = -.infinity
                    for i in 0 ..< blockLength where block[i] == maskId {
                        if confArr[i] >= threshold {
                            block[i] = tokenArr[i]
                            anyCommitted = true
                        }
                        if confArr[i] > bestConf {
                            bestConf = confArr[i]
                            bestIdx = i
                        }
                    }
                    if !anyCommitted, bestIdx >= 0 {
                        block[bestIdx] = tokenArr[bestIdx]
                    }
                } else {
                    let sampledTokens = Self.greedyTokens(logits: draftLogits)
                    eval(sampledTokens)
                    let tokenArr = sampledTokens[0].asArray(Int32.self)

                    for c in cache { _ = c.trim(blockLength) }

                    for i in 0 ..< blockLength where block[i] == maskId {
                        block[i] = tokenArr[i]
                    }
                    break
                }
            }

            // Verify phase — LoRA off, causal, cache update on.
            setLoRAEnabledFast(false)
            let blockTensor = MLXArray(block)[.newAxis, 0...]
            logits = self(blockTensor, cache: cache)
            eval(cache, logits)
            nfe += 1

            let verifyTokens = Self.greedyTokens(logits: logits)
            eval(verifyTokens)
            let arTokens = verifyTokens[0].asArray(Int32.self)

            // Acceptance: longest prefix where verify_tokens[i] == draft_block[i+1],
            // plus one bonus AR token.
            var accepted = 0
            for i in 0 ..< (blockLength - 1) {
                if arTokens[i] == block[i + 1] {
                    accepted += 1
                } else {
                    break
                }
            }
            accepted += 1  // bonus

            let acceptedToks = Array(arTokens.prefix(accepted))
            generated.append(contentsOf: acceptedToks)
            totalGen += accepted

            // Trim cache so that rejected speculative kv entries are discarded.
            let desired = cacheLenBefore + accepted
            for c in cache {
                let trimAmount = max(0, c.offset - desired)
                if trimAmount > 0 {
                    _ = c.trim(trimAmount)
                }
            }

            nextToken = arTokens[accepted - 1]

            if let eos = eosTokenId {
                let eosI32 = Int32(eos)
                if let pos = acceptedToks.firstIndex(of: eosI32) {
                    let truncated = Array(acceptedToks.prefix(pos + 1))
                    generated.removeLast(acceptedToks.count)
                    generated.append(contentsOf: truncated)
                    totalGen = totalGen - accepted + truncated.count
                    onAccept?(promptIds + generated.map { Int($0) }, nfe)
                    break
                }
            }

            onAccept?(promptIds + generated.map { Int($0) }, nfe)

            if totalGen >= maxNewTokens { break }
        }

        let outTokens = promptIds + generated.prefix(maxNewTokens).map { Int($0) }
        return (outTokens, nfe)
    }
}
