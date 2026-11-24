import Foundation
import MLX
import MLXNN

enum Ernie55Language {
    typealias Configuration = Ernie55Configuration
    typealias Head = DeepSeekV4Language.HyperHead

    static func situ(
        gate: MLXArray, up: MLXArray, gateCap: Float, upCap: Float
    ) -> MLXArray {
        let g = gate.asType(.float32)
        let u = up.asType(.float32)
        let boundedGate = gateCap * tanh(g / gateCap) * sigmoid(g)
        let boundedUp = upCap * tanh(u / upCap)
        return (boundedGate * boundedUp).asType(gate.dtype)
    }

    final class Dense: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        let config: Configuration

        init(_ config: Configuration, width: Int) {
            self.config = config
            _gate.wrappedValue = Linear(config.hidden, width, bias: false)
            _up.wrappedValue = Linear(config.hidden, width, bias: false)
            _down.wrappedValue = Linear(width, config.hidden, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(situ(gate: gate(x), up: up(x), gateCap: config.gateCap, upCap: config.upCap))
        }
    }

    final class Router: Module {
        @ParameterInfo var weight: MLXArray
        @ParameterInfo(key: "weight_1") var second: MLXArray
        @ParameterInfo(key: "routed_scaling_factor_param") var scaling: MLXArray
        let topK: Int
        let normalize: Bool

        init(_ config: Configuration) {
            topK = config.topK
            normalize = config.normalizeRouting
            _weight.wrappedValue = .zeros([config.experts, config.hidden])
            _second.wrappedValue = .zeros([config.experts, config.hidden])
            _scaling.wrappedValue = full([config.experts], values: config.routeScale)
        }

        func callAsFunction(_ x: MLXArray, bias: MLXArray) -> (MLXArray, MLXArray) {
            let fp = x.asType(.float32)
            let scores =
                sigmoid(matmul(fp, weight.asType(.float32).T))
                + sigmoid(matmul(fp, second.asType(.float32).T))
            let indices = argPartition(-(scores + bias), kth: topK - 1, axis: -1)[
                .ellipsis, ..<topK]
            var selected = takeAlong(scores, indices, axis: -1)
            if normalize {
                selected = selected / maximum(selected.sum(axis: -1, keepDims: true), 1e-20)
            }
            return (indices, selected * scaling[indices])
        }
    }

    final class Experts: Module, UnaryLayer {
        @ModuleInfo var gate: Router
        @ParameterInfo(key: "e_score_correction_bias") var correction: MLXArray
        @ModuleInfo(key: "fc1_latent_proj") var down: Linear
        @ModuleInfo(key: "switch_mlp") var routed: SwitchGLU
        @ModuleInfo(key: "latent_norm") var norm: RMSNorm
        @ModuleInfo(key: "fc2_latent_proj") var up: Linear
        @ModuleInfo(key: "shared_experts") var shared: Dense

        init(_ config: Configuration) {
            _gate.wrappedValue = Router(config)
            _correction.wrappedValue = .zeros([config.experts])
            _down.wrappedValue = Linear(config.hidden, config.expertLatent, bias: false)
            let gateCap = config.gateCap
            let upCap = config.upCap
            _routed.wrappedValue = SwitchGLU(
                inputDims: config.expertLatent, hiddenDims: config.expertWidth,
                numExperts: config.experts,
                activationProduct: { gate, up in
                    situ(gate: gate, up: up, gateCap: gateCap, upCap: upCap)
                })
            _norm.wrappedValue = RMSNorm(dimensions: config.expertLatent, eps: config.rmsEps)
            _up.wrappedValue = Linear(config.expertLatent, config.hidden, bias: false)
            _shared.wrappedValue = Dense(config, width: config.sharedExperts * config.expertWidth)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (indices, weights) = gate(x, bias: correction)
            let values = routed(down(x), indices).asType(.float32)
            let merged = (values * weights.expandedDimensions(axis: -1)).sum(axis: -2)
            return shared(x) + up(norm(merged.asType(x.dtype)))
        }
    }

    final class Mixer: HyperConnection {
        init(_ config: Configuration) {
            super.init(
                hiddenSize: config.hidden, multiplier: config.hcStreams,
                rmsNormEps: config.rmsEps, eps: config.hcEps,
                sinkhornIterations: config.sinkhornIterations)
        }

        override func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
            let value = x.asType(.float32)
            let flat = value.flattened(start: -2)
            // The sublayer mixers divide by RMS + epsilon after the projection.
            let mixes =
                matmul(flat, fn.asType(.float32).T)
                / (sqrt(square(flat).mean(axis: -1, keepDims: true)) + rmsNormEps)
            let (pre, post, comb) = hyperConnectionSplitSinkhorn(
                mixes: mixes, scale: scale, base: base, multiplier: multiplier,
                sinkhornIterations: sinkhornIterations, eps: eps)
            return (
                (pre.expandedDimensions(axis: -1) * value).sum(axis: -2).asType(x.dtype), post, comb
            )
        }
    }

    final class Layer: Module {
        @ModuleInfo(key: "attn") var attention: Ernie55Attention.Layer
        @ModuleInfo(key: "ffn") var feedForward: UnaryLayer
        @ModuleInfo(key: "attn_norm") var attentionNorm: RMSNorm
        @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
        @ModuleInfo(key: "attn_hc") var attentionMixer: Mixer
        @ModuleInfo(key: "ffn_hc") var ffnMixer: Mixer

        init(_ config: Configuration, index: Int) {
            let ratio = config.compression[index]
            _attention.wrappedValue =
                ratio == -2
                ? Ernie55Attention.MLA(config) : Ernie55Attention.HCA(config, ratio: ratio)
            _feedForward.wrappedValue =
                index < config.denseLayers
                ? Dense(config, width: config.denseWidth) : Experts(config)
            _attentionNorm.wrappedValue = RMSNorm(dimensions: config.hidden, eps: config.rmsEps)
            _ffnNorm.wrappedValue = RMSNorm(dimensions: config.hidden, eps: config.rmsEps)
            _attentionMixer.wrappedValue = Mixer(config)
            _ffnMixer.wrappedValue = Mixer(config)
        }

        func callAsFunction(_ x: MLXArray, cache: CacheList?) -> MLXArray {
            let (query, post, comb) = attentionMixer(x)
            let attended = attention(attentionNorm(query), cache: cache)
            let residual = hyperConnectionExpand(attended, residual: x, post: post, comb: comb)
            let (input, ffnPost, ffnComb) = ffnMixer(residual)
            return hyperConnectionExpand(
                feedForward(ffnNorm(input)), residual: residual, post: ffnPost, comb: ffnComb)
        }
    }

    final class Backbone: Module {
        @ModuleInfo(key: "embed_tokens") var embedding: Embedding
        @ModuleInfo var layers: [Layer]
        @ModuleInfo(key: "hc_head") var head: Head
        @ModuleInfo var norm: RMSNorm
        let config: Configuration

        init(_ config: Configuration) {
            self.config = config
            _embedding.wrappedValue = Embedding(
                embeddingCount: config.vocabulary, dimensions: config.hidden)
            _layers.wrappedValue = (0 ..< config.layers).map { Layer(config, index: $0) }
            _head.wrappedValue = Head(
                hidden: config.hidden, streams: config.hcStreams, rmsEps: config.rmsEps,
                hcEps: config.hcEps)
            _norm.wrappedValue = RMSNorm(dimensions: config.hidden, eps: config.rmsEps)
        }

        func callAsFunction(_ ids: MLXArray, cache: [KVCache]?) -> MLXArray {
            let tokens = ids.ndim == 1 ? ids.expandedDimensions(axis: 0) : ids
            let embeds = embedding(tokens)
            var h = broadcast(
                embeds.expandedDimensions(axis: -2),
                to: [tokens.dim(0), tokens.dim(1), config.hcStreams, config.hidden])
            for (index, layer) in layers.enumerated() {
                h = layer(h, cache: cache?[index] as? CacheList)
            }
            return norm(head(h))
        }
    }

    final class MultiMax: Module {
        @ParameterInfo(key: "multimax_ranges") var ranges: MLXArray
        @ParameterInfo(key: "multimax_ts") var temperatures: MLXArray

        override init() {
            _ranges.wrappedValue = .zeros([4])
            _temperatures.wrappedValue = .zeros([4])
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let logits = x.asType(.float32)
            return logits
                + temperatures[0] * maximum(ranges[0] - logits, 0)
                + temperatures[1] * maximum(logits - ranges[1], 0)
                + temperatures[2] * square(maximum(ranges[2] - logits, 0))
                + temperatures[3] * square(maximum(logits - ranges[3], 0))
        }
    }

    /// The checkpoint's auxiliary tensors for the draft that shares the final decoder layer.
    final class Echo: Module {
        @ModuleInfo var enorm: RMSNorm
        @ModuleInfo var hnorm: RMSNorm
        @ModuleInfo(key: "e_proj") var embedProjection: Linear
        @ModuleInfo(key: "h_proj") var hiddenProjection: Linear
        @ModuleInfo(key: "hc_head") var head: Head
        @ModuleInfo var norm: RMSNorm
        @ParameterInfo(key: "e_score_correction_bias") var correction: MLXArray

        init(_ config: Configuration) {
            _enorm.wrappedValue = RMSNorm(dimensions: config.hidden, eps: config.rmsEps)
            _hnorm.wrappedValue = RMSNorm(dimensions: config.hidden, eps: config.rmsEps)
            _embedProjection.wrappedValue = Linear(config.hidden, config.hidden, bias: false)
            _hiddenProjection.wrappedValue = Linear(config.hidden, config.hidden, bias: false)
            _head.wrappedValue = Head(
                hidden: config.hidden, streams: config.hcStreams, rmsEps: config.rmsEps,
                hcEps: config.hcEps)
            _norm.wrappedValue = RMSNorm(dimensions: config.hidden, eps: config.rmsEps)
            _correction.wrappedValue = .zeros([config.experts])
        }
    }
}

/// ERNIE 5.5 Flash Base text completion.
public final class Ernie55Model: Module, LLMModel, KVCacheDimensionProvider,
    UnwrappedLanguageModelWeights
{
    @ModuleInfo var model: Ernie55Language.Backbone
    @ModuleInfo(key: "lm_head") var head: Linear
    @ModuleInfo var multimax: Ernie55Language.MultiMax
    @ModuleInfo var echo: Ernie55Language.Echo?
    public let config: Ernie55Configuration
    public var vocabularySize: Int { config.vocabulary }
    public var kvHeads: [Int] { Array(repeating: 1, count: config.layers) }
    public var loraLayers: [Module] { model.layers }

    public convenience init(_ config: Ernie55Configuration) {
        self.init(config, retainEcho: false)
    }

    init(_ config: Ernie55Configuration, retainEcho: Bool) {
        self.config = config
        _model.wrappedValue = Ernie55Language.Backbone(config)
        _head.wrappedValue = Linear(config.hidden, config.vocabulary, bias: false)
        _multimax.wrappedValue = Ernie55Language.MultiMax()
        if retainEcho { _echo.wrappedValue = Ernie55Language.Echo(config) }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        multimax(head(model(inputs, cache: cache).asType(.float32)))
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        config.compression.map { ratio in
            if ratio == -2 {
                return CacheList(
                    MaterializedKVCache(KVCacheSimple()), MaterializedKVCache(KVCacheSimple()))
            }
            return CacheList(
                MaterializedKVCache(RotatingKVCache(maxSize: config.window)),
                DeepSeekV4PoolingCache(ratio: ratio))
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Ernie55Checkpoint.sanitize(Self.unwrapLanguageModelWeights(weights), config: config)
    }

    public func quantizationPredicate(path: String, module: Module) -> QuantizationDecision {
        if path.contains(".switch_mlp.") { return .useDefault }
        if path.contains(".shared_experts.") || path.hasSuffix("fc1_latent_proj")
            || path.hasSuffix("fc2_latent_proj") || path == "echo.e_proj" || path == "echo.h_proj"
        {
            return .parameters(.init(groupSize: 32, bits: 8, mode: .mxfp8))
        }
        return .skip
    }
}
