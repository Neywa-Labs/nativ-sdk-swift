import Foundation
import MLX
import MLXNN

final class MistralLarge4MoE: Module, UnaryLayer {
    @ModuleInfo var gate: Linear
    @ModuleInfo(key: "switch_mlp") var experts: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: SwiGLUMLP?

    private let config: MistralLarge4Configuration.Text

    init(_ config: MistralLarge4Configuration.Text) {
        self.config = config
        _gate.wrappedValue = Linear(config.hiddenSize, config.experts, bias: false)
        _experts.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize, hiddenDims: config.expertSize, numExperts: config.experts)
        if config.sharedExperts > 0 {
            _sharedExperts.wrappedValue = SwiGLUMLP(
                inputDimensions: config.hiddenSize,
                hiddenDimensions: config.expertSize * config.sharedExperts)
        }
    }

    func route(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let logits = x.asType(.float32).matmul(gate.weight.asType(.float32).transposed())
        let scores = softmax(logits, axis: -1, precise: true)
        var choice = scores
        if config.groups > 1 {
            let width = config.experts / config.groups
            let shape = Array(scores.shape.dropLast())
            let grouped = scores.reshaped(shape + [config.groups, width])
            let groups = sorted(grouped, axis: -1)[.ellipsis, (width - min(2, width))...].sum(
                axis: -1)
            let best = argPartition(-groups, kth: config.routeGroups - 1, axis: -1)[
                .ellipsis, ..<config.routeGroups]
            let include = any(
                best.expandedDimensions(axis: -1) .== MLXArray(0 ..< config.groups), axis: -2)
            let keep = broadcast(
                include.expandedDimensions(axis: -1), to: shape + [config.groups, width]
            )
            .reshaped(scores.shape)
            choice = MLX.where(keep, scores, -Float.infinity)
        }
        let indices = sorted(
            argPartition(-choice, kth: config.routes - 1, axis: -1)[.ellipsis, ..<config.routes],
            axis: -1)
        let chosen = takeAlong(scores, indices, axis: -1)
        return (
            indices, chosen / (chosen.sum(axis: -1, keepDims: true) + 1e-20) * config.routedScale
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = route(x)
        let routed = (experts(x, indices).asType(.float32) * weights.expandedDimensions(axis: -1))
            .asType(x.dtype)
        // Sum in ascending expert order, with a rounding boundary for every route.
        var output = routed[.ellipsis, 0, 0...]
        for route in 1 ..< config.routes {
            output = output + routed[.ellipsis, route, 0...]
        }
        if let sharedExperts { output = output + sharedExperts(x) }
        return output
    }
}

final class MistralLarge4DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: MistralLarge4Attention
    @ModuleInfo(key: "input_layernorm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var ffnNorm: RMSNorm
    @ModuleInfo var mlp: Module

    init(_ config: MistralLarge4Configuration.Text, layer: Int) {
        _attention.wrappedValue = MistralLarge4Attention(config, layer: layer)
        _attentionNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.eps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.eps)
        _mlp.wrappedValue =
            layer < config.denseLayers
            ? SwiGLUMLP(
                inputDimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
            : MistralLarge4MoE(config)
    }

    func callAsFunction(
        _ x: MLXArray, cache: MistralLarge4Cache?, context: inout MistralLarge4AttentionContext
    ) -> MLXArray {
        let residual = x + attention(attentionNorm(x), cache: cache, context: &context)
        return residual + (mlp as! UnaryLayer)(ffnNorm(residual))
    }
}

final class MistralLarge4Backbone: Module {
    @ModuleInfo(key: "embed_tokens") var embeddings: Embedding
    @ModuleInfo(key: "rotary_emb") var rotary: MistralLarge4LeRoPE
    let layers: [MistralLarge4DecoderLayer]
    let norm: RMSNorm

    init(_ config: MistralLarge4Configuration.Text) {
        _embeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _rotary.wrappedValue = MistralLarge4LeRoPE(config)
        layers = (0 ..< config.layers).map { MistralLarge4DecoderLayer(config, layer: $0) }
        norm = RMSNorm(dimensions: config.hiddenSize, eps: config.eps)
    }

    func callAsFunction(
        _ tokens: MLXArray, embeddings: MLXArray?, cache: [KVCache]?, mask: MLXArray?
    ) -> MLXArray {
        var x = embeddings ?? self.embeddings(tokens)
        let (batch, length) = (x.dim(0), x.dim(1))
        let offset = cache?.first?.offset ?? 0
        let keyLength = offset + length
        let positions = broadcast(
            MLXArray(offset ..< keyLength).reshaped(1, length), to: [batch, length])
        var causal = broadcast(
            createCausalMask(n: length, offset: offset).reshaped(1, 1, length, keyLength),
            to: [batch, 1, length, keyLength])
        if let mask, mask.dim(-1) >= keyLength {
            if mask.ndim == 2 {
                let padding = mask[.ellipsis, ..<keyLength].asType(.bool)
                    .reshaped(mask.dim(0), 1, 1, keyLength)
                causal = logicalAnd(causal, padding)
            } else if mask.ndim == 4 {
                let padding = mask[0..., 0..., (-length)..., ..<keyLength].asType(.bool)
                causal = logicalAnd(causal, padding)
            }
        }
        var context = MistralLarge4AttentionContext(
            positions: positions, rotation: rotary(positions, dtype: x.dtype), causal: causal)
        for (index, layer) in layers.enumerated() {
            x = layer(x, cache: cache?[index] as? MistralLarge4Cache, context: &context)
        }
        return norm(x)
    }
}

public final class MistralLarge4TextModel: Module, LLMModel {
    let config: MistralLarge4Configuration
    let model: MistralLarge4Backbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public var vocabularySize: Int { config.text.vocabularySize }
    public var toolCallFormat: ToolCallFormat? { .mistral }
    public var reasoningConfig: ReasoningConfig? {
        .init(
            startDelimiter: "[THINK]", endDelimiter: "[/THINK]", promptStrategy: .none,
            isSpecialToken: true)
    }
    public var loraLayers: [Module] { model.layers }

    public init(_ config: MistralLarge4Configuration) {
        self.config = config
        model = MistralLarge4Backbone(config.text)
        if !config.text.tiedEmbeddings {
            _lmHead.wrappedValue = Linear(
                config.text.hiddenSize, config.text.vocabularySize, bias: false)
        }
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        if let configuration {
            guard configuration.strategy.identifier == .fullPrecision,
                configuration.capacity == nil
            else {
                throw ModelExecutionError.invalidOptions(
                    "Mistral Large 4 uses a growing, full-precision latent and indexer cache")
            }
        }
        return config.text.kvSources.enumerated().map { layer, source in
            MistralLarge4Cache(
                ownsKV: layer == source, ownsIndexer: config.text.fullIndexers[layer])
        }
    }

    func logits(
        _ tokens: MLXArray, cache: [KVCache]?, embeddings: MLXArray? = nil, mask: MLXArray? = nil
    ) -> MLXArray {
        let hidden = model(tokens, embeddings: embeddings, cache: cache, mask: mask)
        let logits = lmHead.map { $0(hidden) } ?? model.embeddings.asLinear(hidden)
        let cap = config.text.softCap
        return cap > 0 ? tanh(logits / cap) * cap : logits
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokens = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs
        return logits(tokens, cache: cache)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        try prepare(input.text, cache: cache, embeddings: nil, prefill: prefill)
    }

    func prepare(
        _ input: LMInput.Text, cache: [KVCache], embeddings: MLXArray?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        let tokens =
            input.tokens.ndim == 1 ? input.tokens.expandedDimensions(axis: 0) : input.tokens
        let mask = input.mask.map { $0.ndim == 1 ? $0.expandedDimensions(axis: 0) : $0 }
        let total = tokens.dim(1)
        let processed = try prefill.forEachChunk(total: total) { range in
            let features = embeddings?[0..., range, 0...]
            _ = model(tokens[0..., range], embeddings: features, cache: cache, mask: mask)
            asyncEval(cache)
        }
        if processed > 0 { eval(cache) }
        let result = logits(
            tokens[0..., processed...], cache: cache,
            embeddings: embeddings?[0..., processed..., 0...],
            mask: mask)
        prefill.progress?(total, total)
        return .logits(.init(logits: result))
    }
}

public final class MistralLarge4: Module, VLMModel {
    let config: MistralLarge4Configuration
    @ModuleInfo(key: "language_model") var language: MistralLarge4TextModel
    @ModuleInfo(key: "vision_tower") var vision: PixtralVision.VisionModel
    @ModuleInfo(key: "multi_modal_projector") var projector: Mistral3MultiModalProjector

    public var vocabularySize: Int { language.vocabularySize }
    public var toolCallFormat: ToolCallFormat? { language.toolCallFormat }
    public var reasoningConfig: ReasoningConfig? { language.reasoningConfig }
    public var loraLayers: [Module] { language.loraLayers }

    public init(_ config: MistralLarge4Configuration) throws {
        self.config = config
        guard let vision = config.vision else {
            throw ModelExecutionError.invalidOptions(
                "Mistral Large 4 VLM requires a vision configuration")
        }
        _language.wrappedValue = MistralLarge4TextModel(config)
        _vision.wrappedValue = PixtralVision.VisionModel(vision, useFusedAttention: true)
        _projector.wrappedValue = Mistral3MultiModalProjector(
            visionHiddenSize: vision.hiddenSize, textHiddenSize: config.text.hiddenSize,
            patchSize: vision.patchSize, spatialMergeSize: config.spatialMergeSize,
            bias: config.projectorBias, eps: config.text.eps, featureDType: nil)
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try language.newCache(configuration: configuration)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        language(inputs, cache: cache)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        var features: MLXArray?
        if let image = input.image {
            let tokens = input.text.tokens
            let ids = tokens.ndim == 1 ? tokens.expandedDimensions(axis: 0) : tokens
            features = try inputEmbeddings(ids, pixels: image.pixels, frames: image.frames)
        }
        return try language.prepare(
            input.text, cache: cache, embeddings: features, prefill: prefill)
    }

    func inputEmbeddings(_ tokens: MLXArray, pixels: MLXArray, frames: [THW]?) throws -> MLXArray {
        let pixels = pixels.ndim == 3 ? pixels.expandedDimensions(axis: 0) : pixels
        let alignment = config.vision!.patchSize * config.spatialMergeSize
        guard tokens.ndim == 2, tokens.dim(0) == 1, pixels.ndim == 4,
            pixels.dim(0) == 1, pixels.dim(1) == 3, alignment > 0,
            pixels.dim(2) > 0, pixels.dim(3) > 0, pixels.dim(2).isMultiple(of: alignment),
            pixels.dim(3).isMultiple(of: alignment)
        else {
            throw VLMError.processing(
                "Mistral Large 4 requires one image aligned to its merged-patch grid")
        }
        let sizes = frames?.map { ($0.h, $0.w) } ?? [(pixels.dim(2), pixels.dim(3))]
        guard sizes.count == 1, sizes[0].0 == pixels.dim(2), sizes[0].1 == pixels.dim(3) else {
            throw VLMError.processing("Mistral Large 4 image sizes must match the input pixels")
        }
        let (_, _, states) = vision(pixels.transposed(0, 2, 3, 1), outputHiddenStates: true)
        let layer = config.visionFeatureLayer
        guard let states, states.indices.contains(layer >= 0 ? layer : states.count + layer) else {
            throw VLMError.processing(
                "Mistral Large 4 vision_feature_layer is outside the vision tower")
        }
        let projected = projector(
            states[layer >= 0 ? layer : states.count + layer], imageSizes: sizes)
        let locations = tokens[0].asArray(Int.self).enumerated().compactMap { index, token in
            token == config.imageTokenIndex ? index : nil
        }
        guard !locations.isEmpty, locations.count == projected.dim(1) else {
            throw VLMError.processing(
                "Mistral Large 4 image tokens and projected patches must agree")
        }
        let embeddings = language.model.embeddings(tokens)
        let indices = broadcast(
            MLXArray(locations).reshaped(1, locations.count, 1), to: projected.shape)
        return putAlong(embeddings, indices, values: projected.asType(embeddings.dtype), axis: 1)
    }
}
