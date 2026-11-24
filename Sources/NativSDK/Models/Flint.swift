// Flint architecture: https://huggingface.co/RecursiveSI/flint (Apache-2.0).
import Foundation
import MLX
import MLXNN

public struct FlintConfiguration: Decodable, Sendable {
    var hiddenSize: Int
    var intermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var vocabularySize: Int
    var rmsNormEps: Float
    var maxPositionEmbeddings: Int
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var layerTypes: [String]
    var noRopeLayers: [Int]
    var slidingWindow: Int?
    var attentionBias: Bool
    var denseLayers: [Int]
    var numExperts: Int
    var expertsPerToken: Int
    var moeIntermediateSize: Int
    var sharedIntermediateSize: Int
    var tieWordEmbeddings: Bool
    var embeddingBits: Int?
    private var activation: String
    private var routerFunction: String
    private var softmaxType: String
    private var sparseStep: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
        case layerTypes = "layer_types"
        case noRopeLayers = "no_rope_layer_indices"
        case useSlidingWindow = "use_sliding_window"
        case slidingWindow = "sliding_window"
        case attentionBias = "attention_bias"
        case denseLayers = "mlp_only_layers"
        case numExperts = "num_experts"
        case expertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedIntermediateSize = "shared_expert_intermediate_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case mlxExport = "mlx_export"
        case activation = "hidden_act"
        case routerFunction = "router_score_function"
        case softmaxType = "softmax_type"
        case sparseStep = "decoder_sparse_step"
    }
    private struct Export: Decodable {
        let embeddingBits: Int?
        enum CodingKeys: String, CodingKey { case embeddingBits = "embedding_bits" }
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 49
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 150144
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262144
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        ropeScaling =
            try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
            ?? c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        if let theta = ropeScaling?["rope_theta"]?.asFloat() { ropeTheta = theta }
        if ropeScaling?["type"] == nil, let type = ropeScaling?["rope_type"] {
            ropeScaling?["type"] = type
        }
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "full_attention", count: max(0, hiddenLayers))
        noRopeLayers = try c.decodeIfPresent([Int].self, forKey: .noRopeLayers) ?? []
        let useSliding = try c.decodeIfPresent(Bool.self, forKey: .useSlidingWindow) ?? true
        slidingWindow =
            try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
            ?? (c.contains(.slidingWindow) ? nil : 513)
        if !useSliding { slidingWindow = nil }
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        denseLayers = try c.decodeIfPresent([Int].self, forKey: .denseLayers) ?? []
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 256
        expertsPerToken = try c.decodeIfPresent(Int.self, forKey: .expertsPerToken) ?? 6
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 512
        sharedIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .sharedIntermediateSize) ?? 1024
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        embeddingBits = try c.decodeIfPresent(Export.self, forKey: .mlxExport)?.embeddingBits
        activation = try c.decodeIfPresent(String.self, forKey: .activation) ?? "silu"
        routerFunction = try c.decodeIfPresent(String.self, forKey: .routerFunction) ?? "sigmoid"
        softmaxType = try c.decodeIfPresent(String.self, forKey: .softmaxType) ?? "learnable"
        sparseStep = try c.decodeIfPresent(Int.self, forKey: .sparseStep) ?? 1
    }
}

extension FlintConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        guard hiddenSize > 0, intermediateSize > 0, hiddenLayers > 0,
            attentionHeads > 0, kvHeads > 0, attentionHeads.isMultiple(of: kvHeads),
            headDim > 0, headDim.isMultiple(of: 2), vocabularySize > 0,
            rmsNormEps.isFinite, rmsNormEps > 0, ropeTheta.isFinite, ropeTheta > 0,
            maxPositionEmbeddings > 0, layerTypes.count == hiddenLayers,
            layerTypes.allSatisfy({ ["full_attention", "sliding_attention"].contains($0) }),
            (noRopeLayers + denseLayers).allSatisfy({ (0 ..< hiddenLayers).contains($0) }),
            slidingWindow.map({ $0 > 0 }) ?? true,
            numExperts > 0, (1 ... numExperts).contains(expertsPerToken),
            moeIntermediateSize > 0, sharedIntermediateSize >= 0,
            activation == "silu", routerFunction == "sigmoid", softmaxType == "learnable",
            sparseStep == 1
        else {
            throw ModelExecutionError.invalidOptions("Invalid or unsupported Flint configuration")
        }
        try validateRoPEConfiguration(ropeScaling, context: "Flint.rope_parameters")
    }
}

enum FlintLayers {
    final class Attention: Module {
        @ModuleInfo(key: "q_proj") var query: Linear
        @ModuleInfo(key: "k_proj") var key: Linear
        @ModuleInfo(key: "v_proj") var value: Linear
        @ModuleInfo(key: "o_proj") var output: Linear
        @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
        @ParameterInfo(key: "softmax_offset") var sink: MLXArray
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let window: Int?
        let rope: RoPELayer?
        init(_ config: FlintConfiguration, index: Int) {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            window = config.layerTypes[index] == "sliding_attention" ? config.slidingWindow : nil
            _query.wrappedValue = Linear(
                config.hiddenSize, heads * headDim, bias: config.attentionBias)
            _key.wrappedValue = Linear(
                config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
            _value.wrappedValue = Linear(
                config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
            _output.wrappedValue = Linear(
                heads * headDim, config.hiddenSize, bias: config.attentionBias)
            _queryNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
            _keyNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
            _sink.wrappedValue = MLXArray.zeros([heads])
            rope =
                config.noRopeLayers.contains(index)
                ? nil
                : initializeRope(
                    dims: headDim, base: config.ropeTheta, traditional: false,
                    scalingConfig: config.ropeScaling,
                    maxPositionEmbeddings: config.maxPositionEmbeddings)
        }
        func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
            let mask = createAttentionMask(h: x, cache: cache, windowSize: window)
            var q = queryNorm(query(x).reshaped(x.dim(0), x.dim(1), heads, headDim)).transposed(
                0, 2, 1, 3)
            var k = keyNorm(key(x).reshaped(x.dim(0), x.dim(1), kvHeads, headDim)).transposed(
                0, 2, 1, 3)
            var v = value(x).reshaped(x.dim(0), x.dim(1), kvHeads, headDim).transposed(0, 2, 1, 3)
            if let rope {
                let offset = cache?.ropeOffset
                q = applyRotaryPosition(rope, to: q, offset: offset)
                k = applyRotaryPosition(rope, to: k, offset: offset)
            }
            if let packed = cache as? QuantizedKVCacheProtocol {
                let (keys, values) = packed.updateQuantized(keys: k, values: v)
                // Keep learned sinks on the ordinary SDPA path when a packed cache is requested.
                k = dequantized(
                    keys.0, scales: keys.1, biases: keys.2,
                    groupSize: packed.groupSize, bits: packed.bits, mode: packed.mode
                ).asType(q.dtype)
                v = dequantized(
                    values.0, scales: values.1, biases: values.2,
                    groupSize: packed.groupSize, bits: packed.bits, mode: packed.mode
                ).asType(q.dtype)
            } else if let cache {
                (k, v) = cache.update(keys: k, values: v)
            }
            let attention = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v,
                scale: pow(Float(headDim), -0.5), mask: mask, sinks: sink.asType(q.dtype))
            return output(attention.transposed(0, 2, 1, 3).reshaped(x.dim(0), x.dim(1), -1))
        }
    }
    final class MoE: Module, UnaryLayer {
        @ModuleInfo var gate: Linear
        @ParameterInfo(key: "expert_bias") var expertBias: MLXArray
        @ModuleInfo(key: "switch_mlp") var experts: SwitchGLU
        @ModuleInfo(key: "shared_expert") var shared: Qwen3MLP?
        let topK: Int
        init(_ config: FlintConfiguration) {
            topK = config.expertsPerToken
            _gate.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
            _expertBias.wrappedValue = MLXArray.zeros([config.numExperts])
            _experts.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize, numExperts: config.numExperts)
            if config.sharedIntermediateSize > 0 {
                _shared.wrappedValue = Qwen3MLP(
                    dimensions: config.hiddenSize, hiddenDimensions: config.sharedIntermediateSize)
            }
        }
        func route(_ x: MLXArray) -> (MLXArray, MLXArray) {
            let scores = sigmoid(gate(x).asType(.float32))
            let selection = scores + expertBias.asType(.float32)
            let indices = stopGradient(
                argPartition(selection, kth: -topK, axis: -1)[.ellipsis, (-topK)...])
            let weights = takeAlong(scores, indices, axis: -1)
            return (indices, weights / (weights.sum(axis: -1, keepDims: true) + Float(1e-20)))
        }
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (indices, weights) = route(x)
            let routed = experts(x, indices)
            var output = (routed.asType(.float32) * weights[.ellipsis, .newAxis]).sum(axis: -2)
                .asType(x.dtype)
            if let shared { output = output + shared(x) }
            return output
        }
    }
    final class Layer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
        @ModuleInfo var mlp: Module
        init(_ config: FlintConfiguration, index: Int) {
            _attention.wrappedValue = Attention(config, index: index)
            _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _postNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _mlp.wrappedValue =
                config.denseLayers.contains(index)
                ? Qwen3MLP(dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
                    as Module : MoE(config)
        }
        func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
            let h = x + attention(inputNorm(x), cache: cache)
            return h + (mlp as! UnaryLayer)(postNorm(h))
        }
    }
    final class Model: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        let layers: [Layer]
        let norm: RMSNorm
        init(_ config: FlintConfiguration) {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            layers = (0 ..< config.hiddenLayers).map { Layer(config, index: $0) }
            norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
        func callAsFunction(_ ids: MLXArray, cache: [KVCache]?) -> MLXArray {
            var x = embedTokens(ids)
            for (index, layer) in layers.enumerated() { x = layer(x, cache: cache?[index]) }
            return norm(x)
        }
    }
}

public final class Flint: Module, LLMModel, UnwrappedLanguageModelWeights {
    let config: FlintConfiguration
    let model: FlintLayers.Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    public var vocabularySize: Int { config.vocabularySize }
    public var loraLayers: [Module] { model.layers }
    public init(_ config: FlintConfiguration) {
        self.config = config
        model = FlintLayers.Model(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache).asType(.float32)
        return lmHead.map { $0(hidden) } ?? model.embedTokens.asLinear(hidden)
    }
    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        if configuration?.strategy.identifier == .turboQuant {
            throw ModelExecutionError.invalidOptions(
                "Flint learned attention sinks do not support TurboQuant caches")
        }
        return try model.layers.map {
            try makeHybridAttentionKVCache(
                configuration: configuration,
                slidingWindow: $0.attention.window, usesSlidingWindow: $0.attention.window != nil)
        }
    }
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = Self.unwrapLanguageModelWeights(weights)
        for (key, value) in result where key.hasSuffix(".weight") && value.dtype == .uint8 {
            let path = String(key.dropLast(7))
            if result[path + ".scales"] != nil { result[key] = value.view(dtype: .uint32) }
        }
        result = filterLMHeadWeights(from: result, tiedWordEmbeddings: config.tieWordEmbeddings)
        for (index, layer) in model.layers.enumerated() where layer.mlp is FlintLayers.MoE {
            let prefix = "model.layers.\(index).mlp."
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    let target = prefix + "switch_mlp.\(projection).\(suffix)"
                    let keys = (0 ..< config.numExperts).map {
                        prefix + "experts.\($0).\(projection).\(suffix)"
                    }
                    let arrays = keys.compactMap { result[$0] }
                    guard result[target] == nil, arrays.count == config.numExperts else { continue }
                    result[target] = stacked(arrays)
                    for key in keys { result.removeValue(forKey: key) }
                }
            }
            for alias in ["e_score_correction_bias", "expert_bias"] {
                if let value = result.removeValue(forKey: prefix + "gate." + alias) {
                    result[prefix + "expert_bias"] = value
                }
            }
            if result[prefix + "gate.weight"] != nil, result[prefix + "expert_bias"] == nil {
                result[prefix + "expert_bias"] = MLXArray.zeros([config.numExperts])
            }
        }
        return result
    }
    func checkpointQuantization(_ original: BaseConfiguration.PerLayerQuantization?) throws
        -> BaseConfiguration.PerLayerQuantization?
    {
        guard var quantization = try Self.unwrapLanguageModelQuantization(original) else {
            return nil
        }
        if let bits = config.embeddingBits,
            quantization.perLayerQuantization["model.embed_tokens"] == nil,
            let defaults = quantization.quantization
        {
            let parameters = BaseConfiguration.Quantization(
                groupSize: defaults.groupSize, bits: bits, mode: defaults.mode)
            try parameters.validateForConversion()
            quantization.perLayerQuantization["model.embed_tokens"] = .quantize(parameters)
        }
        return quantization
    }
    public func quantizationPredicate(path: String, module: Module) -> QuantizationDecision {
        path.hasSuffix("mlp.gate") ? .parameters(.init(groupSize: 64, bits: 8)) : .useDefault
    }
    public var toolCallFormat: ToolCallFormat? { .glm4 }
    public var reasoningConfig: ReasoningConfig? {
        .init(
            startDelimiter: "<think>", endDelimiter: "</think>", promptStrategy: .none,
            isSpecialToken: true, implicitEndDelimiters: ["<tool_call>"])
    }
    public func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        ObjectToolArgumentsMessageGenerator()
    }
}
