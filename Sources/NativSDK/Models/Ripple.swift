// Architecture reference: https://huggingface.co/ricursiveAI/Ripple-65B-Preview (Apache-2.0).
import Foundation
import MLX
import MLXNN

public struct RippleConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    enum AttentionKind: String, Decodable, Sendable {
        case full = "full_attention"
        case sliding = "sliding_attention"
    }

    let hiddenSize: Int
    let intermediateSize: Int
    let layerTypes: [AttentionKind]
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let vocabularySize: Int
    let rmsNormEps: Float
    let ropeTheta: Double
    let slidingWindow: Int?
    let useGatedAttention: Bool
    let gateFullPrecision: Bool
    let useValueResidual: Bool
    let clampThreshold: Float?
    let embeddingScale: Float?
    let tieWordEmbeddings: Bool
    private let hiddenLayers: Int
    private let activation: String

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case layerTypes = "layer_types"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
        case useGatedAttention = "use_gated_attention"
        case gateFullPrecision = "gate_full_precision"
        case useValueResidual = "use_valres"
        case clampThreshold = "ffn_clamp_threshold"
        case embeddingScale = "embed_scale"
        case tieWordEmbeddings = "tie_word_embeddings"
        case activation = "hidden_act"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 5120
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 40960
        let count = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 80
        hiddenLayers = count
        layerTypes =
            try c.decodeIfPresent([AttentionKind].self, forKey: .layerTypes)
            ?? (0 ..< max(0, count)).map { $0 % 5 == 4 ? .full : .sliding }
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 80
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 100352
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Double.self, forKey: .ropeTheta) ?? 10000
        slidingWindow =
            c.contains(.slidingWindow)
            ? try c.decodeIfPresent(Int.self, forKey: .slidingWindow) : 1024
        useGatedAttention = try c.decodeIfPresent(Bool.self, forKey: .useGatedAttention) ?? true
        gateFullPrecision = try c.decodeIfPresent(Bool.self, forKey: .gateFullPrecision) ?? true
        useValueResidual = try c.decodeIfPresent(Bool.self, forKey: .useValueResidual) ?? false
        clampThreshold = try c.decodeIfPresent(Float.self, forKey: .clampThreshold)
        embeddingScale = try c.decodeIfPresent(Float.self, forKey: .embeddingScale)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        activation = try c.decodeIfPresent(String.self, forKey: .activation) ?? "silu"
        try validateModelConfiguration()
    }

    public func validateModelConfiguration() throws {
        guard hiddenSize > 0, intermediateSize > 0, hiddenLayers > 0,
            layerTypes.count == hiddenLayers, vocabularySize > 0,
            attentionHeads > 0, kvHeads > 0, attentionHeads.isMultiple(of: kvHeads),
            headDim > 0, headDim.isMultiple(of: 2), activation == "silu",
            rmsNormEps.isFinite, rmsNormEps > 0, ropeTheta.isFinite, ropeTheta > 0,
            slidingWindow.map({ $0 > 0 }) ?? true,
            clampThreshold.map({ $0.isFinite && $0 >= 0 }) ?? true,
            embeddingScale.map(\.isFinite) ?? true
        else {
            throw ModelExecutionError.invalidOptions("Invalid or unsupported Ripple configuration")
        }
    }
}

private enum RippleLayers {
    static let normalize = compile(shapeless: true) { x, weight, eps in
        let hidden = x.asType(.float32)
        let inverseRMS = rsqrt((hidden * hidden).mean(axis: -1, keepDims: true) + eps)
        return (hidden * inverseRMS).asType(x.dtype) * weight
    }

    static let interpolate = compile(shapeless: true) { (x, residual, weight) in
        let blend = sigmoid(weight.asType(.float32)).asType(x.dtype).asType(.float32)
        let seed = residual.asType(x.dtype).asType(.float32)
        let end = x.asType(.float32)
        let delta = end - seed
        return which(blend .< 0.5, seed + blend * delta, end + (blend - 1) * delta).asType(x.dtype)
    }

    static let applySwiGLU = compile(shapeless: true) { (gate, up) in
        let fp32 = gate.asType(.float32)
        return (fp32 * sigmoid(fp32)).asType(gate.dtype) * up
    }

    static let applyClippedSwiGLU = compile(shapeless: true) { gate, up, limit in
        applySwiGLU(minimum(gate, limit), minimum(maximum(up, -limit), limit))
    }

    static let rotate = compile(shapeless: true) { first, second, angles in
        let c = cos(angles).asType(first.dtype).asType(.float32)
        let s = sin(angles).asType(first.dtype).asType(.float32)
        let a = first.asType(.float32)
        let b = second.asType(.float32)
        return concatenated([a * c - b * s, b * c + a * s], axis: -1).asType(first.dtype)
    }

    final class Norm: Module {
        let weight: MLXArray
        let epsilon: Float

        init(_ dimensions: Int, epsilon: Float) {
            weight = MLXArray.ones([dimensions])
            self.epsilon = epsilon
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            // The reference rounds the normalized activation before applying the weight.
            normalize(x, weight, MLXArray(epsilon))
        }
    }

    struct RotaryEmbedding {
        let inverseFrequencies: MLXArray

        init(dimensions: Int, base: Double) {
            let periods = stride(from: 0, to: dimensions, by: 2).map {
                Float(pow(base, Double($0) / Double(dimensions)))
            }
            inverseFrequencies = 1 / MLXArray(periods)
        }

        func angles(length: Int, offset: RoPEOffset?) -> MLXArray {
            let positions = MLXArray(0 ..< length).asType(.float32)
            let start =
                switch offset {
                case .batch(let offsets): offsets.asType(.float32).reshaped(-1, 1)
                case .scalar(let offset): MLXArray(Float(offset)).reshaped(1, 1)
                case nil: MLXArray.zeros([1, 1])
                }
            return (positions + start)[0..., .newAxis, 0..., .newAxis] * inverseFrequencies
        }

        func apply(_ x: MLXArray, angles: MLXArray) -> MLXArray {
            let (first, second) = split(x, axis: -1)
            // Unlike fused RoPE, the trigonometric values are rounded before rotation.
            return rotate(first, second, angles)
        }
    }

    final class Attention: Module {
        @ModuleInfo(key: "q_proj") var query: Linear
        @ModuleInfo(key: "k_proj") var key: Linear
        @ModuleInfo(key: "v_proj") var value: Linear
        @ModuleInfo(key: "o_proj") var output: Linear
        @ModuleInfo(key: "g_proj") var gate: Linear?
        @ModuleInfo(key: "q_norm") var queryNorm: Norm
        @ModuleInfo(key: "k_norm") var keyNorm: Norm
        @ParameterInfo(key: "value_res_lambda") var valueBlend: MLXArray?

        let window: Int?
        let rotary: RotaryEmbedding?
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let gateFullPrecision: Bool

        init(_ config: RippleConfiguration, kind: RippleConfiguration.AttentionKind) {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)
            gateFullPrecision = config.gateFullPrecision
            window = kind == .sliding ? config.slidingWindow : nil
            rotary =
                kind == .sliding
                ? RotaryEmbedding(dimensions: headDim, base: config.ropeTheta) : nil
            _query.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
            _key.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
            _value.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
            _output.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
            _queryNorm.wrappedValue = Norm(headDim, epsilon: config.rmsNormEps)
            _keyNorm.wrappedValue = Norm(headDim, epsilon: config.rmsNormEps)
            if config.useGatedAttention {
                _gate.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
            }
            if config.useValueResidual { _valueBlend.wrappedValue = MLXArray.zeros([1]) }
        }

        func callAsFunction(_ x: MLXArray, cache: KVCache?, valueResidual: MLXArray?) -> MLXArray {
            let (batch, length) = (x.dim(0), x.dim(1))
            let valueInput =
                if let valueResidual, let valueBlend {
                    interpolate(x, valueResidual, valueBlend)
                } else { x }
            var q = queryNorm(query(x).reshaped(batch, length, heads, headDim)).transposed(
                0, 2, 1, 3)
            var k = keyNorm(key(x).reshaped(batch, length, kvHeads, headDim)).transposed(0, 2, 1, 3)
            let v = value(valueInput).reshaped(batch, length, kvHeads, headDim).transposed(
                0, 2, 1, 3)
            if let rotary {
                let angles = rotary.angles(length: length, offset: cache?.ropeOffset)
                q = rotary.apply(q, angles: angles)
                k = rotary.apply(k, angles: angles)
            }
            let mask = createAttentionMask(h: x, cache: cache, windowSize: window)
            var attention = attentionWithCacheUpdate(
                queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask
            ).transposed(0, 2, 1, 3).reshaped(batch, length, -1)
            if let gate {
                let logits = gate(x)
                let probabilities = sigmoid(gateFullPrecision ? logits.asType(.float32) : logits)
                attention = attention * probabilities.asType(attention.dtype)
            }
            return output(attention)
        }
    }

    final class MLP: Module {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        let clampThreshold: Float?

        init(_ config: RippleConfiguration) {
            clampThreshold = config.clampThreshold
            _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
            _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
            _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let gate = gate(x)
            let up = up(x)
            if let limit = clampThreshold, limit > 0 {
                return down(applyClippedSwiGLU(gate, up, MLXArray(limit, dtype: gate.dtype)))
            }
            return down(applySwiGLU(gate, up))
        }
    }

    final class Layer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "input_layernorm") var inputNorm: Norm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: Norm
        @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedForwardNorm: Norm
        @ModuleInfo(key: "post_feedforward_layernorm") var postFeedForwardNorm: Norm
        let mlp: MLP

        init(_ config: RippleConfiguration, kind: RippleConfiguration.AttentionKind) {
            _attention.wrappedValue = Attention(config, kind: kind)
            _inputNorm.wrappedValue = Norm(config.hiddenSize, epsilon: config.rmsNormEps)
            _postAttentionNorm.wrappedValue = Norm(config.hiddenSize, epsilon: config.rmsNormEps)
            _preFeedForwardNorm.wrappedValue = Norm(config.hiddenSize, epsilon: config.rmsNormEps)
            _postFeedForwardNorm.wrappedValue = Norm(config.hiddenSize, epsilon: config.rmsNormEps)
            mlp = MLP(config)
        }

        func callAsFunction(_ x: MLXArray, cache: KVCache?, valueResidual: MLXArray?) -> MLXArray {
            let h =
                x
                + postAttentionNorm(
                    attention(inputNorm(x), cache: cache, valueResidual: valueResidual))
            return h + postFeedForwardNorm(mlp(preFeedForwardNorm(h)))
        }
    }

    final class Model: Module {
        @ModuleInfo(key: "embed_tokens") var embeddings: Embedding
        let layers: [Layer]
        let norm: Norm
        let embeddingScale: Float?
        let useValueResidual: Bool

        init(_ config: RippleConfiguration) {
            _embeddings.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            layers = config.layerTypes.map { Layer(config, kind: $0) }
            norm = Norm(config.hiddenSize, epsilon: config.rmsNormEps)
            embeddingScale = config.embeddingScale
            useValueResidual = config.useValueResidual
        }

        func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
            var h = embeddings(tokens)
            if let embeddingScale {
                h = (h.asType(.float32) * embeddingScale).asType(h.dtype)
            }
            let valueResidual = useValueResidual ? h : nil
            for (index, layer) in layers.enumerated() {
                h = layer(h, cache: cache?[index], valueResidual: valueResidual)
            }
            return norm(h)
        }
    }
}

public final class Ripple: Module, LLMModel, UnwrappedLanguageModelWeights {
    private let config: RippleConfiguration
    private let model: RippleLayers.Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: RippleConfiguration) {
        self.config = config
        model = RippleLayers.Model(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public var vocabularySize: Int { config.vocabularySize }
    public var loraLayers: [Module] { model.layers }
    public var toolCallFormat: ToolCallFormat? { .json }
    public var reasoningConfig: ReasoningConfig? {
        .init(
            startDelimiter: "<think>", endDelimiter: "</think>", promptStrategy: .none,
            implicitEndDelimiters: ["<tool_call>"])
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        return lmHead.map { $0(hidden) } ?? model.embeddings.asLinear(hidden)
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try model.layers.map { layer in
            let window = layer.attention.window
            return try makeHybridAttentionKVCache(
                configuration: configuration, slidingWindow: window,
                usesSlidingWindow: window != nil)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        filterLMHeadWeights(
            from: Self.unwrapLanguageModelWeights(weights),
            tiedWordEmbeddings: config.tieWordEmbeddings)
    }
}
