// Architecture adapted from KRAFTON's Raon-1 Mini reference (Apache-2.0).
import Foundation
import MLX
import MLXNN

private enum RaonLayers {
    static let situ = compile(shapeless: true) { inputs in
        let (gate, value, gateBeta, valueBeta) = (inputs[0], inputs[1], inputs[2], inputs[3])
        let g = gate.asType(.float32)
        let v = value.asType(.float32)
        let activated = gateBeta * tanh(g / gateBeta) * sigmoid(g)
        let bounded = valueBeta * tanh(v / valueBeta)
        return [(activated * bounded).asType(value.dtype)]
    }

    static func activation(_ config: Raon1MiniConfiguration)
        -> @Sendable (MLXArray, MLXArray) -> MLXArray
    {
        let gateBeta = config.gateBeta
        let valueBeta = config.valueBeta
        return { gate, value in
            situ([gate, value, MLXArray(gateBeta), MLXArray(valueBeta)])[0]
        }
    }

    struct Chart {
        let basis: MLXArray
        let identity: MLXArray
        let center: MLXArray

        init(_ streams: Int) {
            let basisValues: [Float] = (0 ..< streams).flatMap { row in
                (0 ..< streams - 1).map { column in
                    let n = column + 1
                    let numerator = row < n ? 1 : row == n ? -Double(n) : 0
                    return Float(numerator / sqrt(Double(n * (n + 1))))
                }
            }
            basis = MLXArray(basisValues, [streams, streams - 1])
            identity = eye(streams)
            center = (identity + 1 / Float(streams)) * 0.5
        }
    }

    final class ResidualChart: Module {
        @ParameterInfo(key: "phi_z") var phi: MLXArray
        @ParameterInfo(key: "alpha_z") var alpha: MLXArray
        @ParameterInfo(key: "b_z") var bias: MLXArray
        let eta = MLXArray(Float(0))

        init(_ config: Raon1MiniConfiguration) {
            let side = config.streams - 1
            _phi.wrappedValue = .zeros([side * side, config.streams * config.hiddenSize])
            _alpha.wrappedValue = MLXArray(Float(1))
            _bias.wrappedValue = .zeros([side * side])
        }
    }

    final class StreamMixer: Module {
        @ParameterInfo(key: "phi_pre") var phiPre: MLXArray
        @ParameterInfo(key: "phi_post") var phiPost: MLXArray
        @ParameterInfo(key: "alpha_pre") var alphaPre: MLXArray
        @ParameterInfo(key: "alpha_post") var alphaPost: MLXArray
        @ParameterInfo(key: "b_pre") var biasPre: MLXArray
        @ParameterInfo(key: "b_post") var biasPost: MLXArray
        let res: ResidualChart
        let streams: Int
        let epsilon: Float
        let postFactor: Float

        init(_ config: Raon1MiniConfiguration) {
            streams = config.streams
            epsilon = config.selectorEpsilon
            postFactor = config.postFactor
            let shape = [streams, streams * config.hiddenSize]
            _phiPre.wrappedValue = .zeros(shape)
            _phiPost.wrappedValue = .zeros(shape)
            _alphaPre.wrappedValue = MLXArray(Float(1))
            _alphaPost.wrappedValue = MLXArray(Float(1))
            _biasPre.wrappedValue = .zeros([streams])
            _biasPost.wrappedValue = .zeros([streams])
            res = ResidualChart(config)
        }

        func routes(_ rows: MLXArray, chart: Chart)
            -> (pre: MLXArray, post: MLXArray, residual: MLXArray)
        {
            let flat = rows.flattened(start: -2)
            let selectors = concatenated([phiPre, phiPost, res.phi]).asType(flat.dtype)
            let normalized = rsqrt(
                square(flat.asType(.float32)).mean(axis: -1, keepDims: true) + epsilon)
            let projection = matmul(flat, selectors.T).asType(.float32) * normalized
            let parts = split(projection, indices: [streams, 2 * streams], axis: -1)
            let pre = sigmoid(alphaPre.asType(.float32) * parts[0] + biasPre.asType(.float32))
            let post =
                postFactor
                * sigmoid(alphaPost.asType(.float32) * parts[1] + biasPost.asType(.float32))
            let z = res.alpha.asType(.float32) * parts[2] + res.bias.asType(.float32)
            let error = matmul(
                matmul(chart.basis, z.reshaped(-1, streams - 1, streams - 1)), chart.basis.T)
            let radius = maximum((-error / chart.center).max(axes: [-2, -1], keepDims: true), 0)
            let gamma = sigmoid(res.eta.asType(.float32))
            let residual =
                (1 - gamma) * chart.identity + gamma * (chart.center + error / (1 + radius))
            return (pre, post, residual)
        }

        static func aggregate(_ rows: MLXArray, pre: MLXArray) -> MLXArray {
            (rows.asType(.float32) * pre[.ellipsis, .newAxis]).sum(axis: -2).asType(rows.dtype)
        }

        static func mix(_ rows: MLXArray, branch: MLXArray, post: MLXArray, residual: MLXArray)
            -> MLXArray
        {
            (matmul(residual, rows.asType(.float32))
                + post[.ellipsis, .newAxis] * branch.asType(.float32)[0..., .newAxis, 0...])
                .asType(rows.dtype)
        }
    }

    final class DeltaAttention: Module {
        @ModuleInfo(key: "q_proj") var query: Linear
        @ModuleInfo(key: "k_proj") var key: Linear
        @ModuleInfo(key: "v_proj") var value: Linear
        @ModuleInfo(key: "g_proj") var gate: Linear
        @ModuleInfo(key: "qkv_conv1d") var convolution: Conv1d
        @ModuleInfo(key: "f_a_proj") var decayA: Linear
        @ModuleInfo(key: "f_b_proj") var decayB: Linear
        @ModuleInfo(key: "b_proj") var writeGate: Linear
        @ModuleInfo(key: "rda_w_proj") var writeModulation: Linear
        @ModuleInfo(key: "o_norm") var outputNorm: RMSNorm
        @ModuleInfo(key: "o_proj") var output: Linear
        @ParameterInfo(key: "A_log") var aLog: MLXArray
        @ParameterInfo(key: "dt_bias") var dtBias: MLXArray

        let heads: Int
        let headDim: Int
        let width: Int
        let scale: Float
        let historySize: Int
        let epsilon: Float
        let lowerBound: Float?

        init(_ config: Raon1MiniConfiguration) {
            heads = config.recurrentHeads
            headDim = config.recurrentHeadDim
            width = heads * headDim
            scale = pow(Float(headDim), -0.5)
            historySize = config.delta.kernelSize - 1
            epsilon = config.normEpsilon
            lowerBound = config.delta.lowerBound
            let hidden = config.hiddenSize
            _query.wrappedValue = Linear(hidden, width, bias: false)
            _key.wrappedValue = Linear(hidden, width, bias: false)
            _value.wrappedValue = Linear(hidden, width, bias: false)
            _gate.wrappedValue = Linear(hidden, width, bias: false)
            _convolution.wrappedValue = Conv1d(
                inputChannels: 3 * width, outputChannels: 3 * width,
                kernelSize: config.delta.kernelSize, groups: 3 * width, bias: false)
            _decayA.wrappedValue = Linear(hidden, config.decayRank, bias: false)
            _decayB.wrappedValue = Linear(config.decayRank, width, bias: false)
            _writeGate.wrappedValue = Linear(hidden, heads, bias: false)
            _writeModulation.wrappedValue = Linear(hidden, heads, bias: false)
            _outputNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: epsilon)
            _output.wrappedValue = Linear(width, hidden, bias: false)
            _aLog.wrappedValue = .zeros([heads], dtype: .float32)
            _dtBias.wrappedValue = .zeros([width], dtype: .float32)
        }

        func callAsFunction(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
            let (batch, length) = (x.dim(0), x.dim(1))
            let shape = [batch, length, heads, headDim]
            let mask = cache?.makeMask(N: length)
            var qkv = concatenated([query(x), key(x), value(x)], axis: -1)
            if let mask { qkv = MLX.where(mask[.ellipsis, .newAxis], qkv, 0) }
            let history =
                cache?[0] ?? MLXArray.zeros([batch, historySize, 3 * width], dtype: x.dtype)
            let signal = concatenated([history, qkv], axis: 1)
            if let cache {
                // Save the raw projection window before the convolution and write gate.
                if let lengths = cache.currentLengths {
                    let positions =
                        clip(lengths, min: 0, max: length)[0..., .newAxis]
                        + MLXArray(0 ..< historySize)
                    cache[0] = takeAlong(signal, positions[.ellipsis, .newAxis], axis: 1)
                } else {
                    cache[0] = contiguous(signal[0..., length..., 0...])
                }
            }
            let convolved = conv1d(
                signal.asType(.float32), convolution.weight.asType(.float32), groups: 3 * width)
            let parts = split(silu(convolved).asType(x.dtype), parts: 3, axis: -1)
            func normalize(_ t: MLXArray) -> MLXArray {
                let fp = t.reshaped(shape).asType(.float32)
                return (fp / maximum(sqrt(square(fp).sum(axis: -1, keepDims: true)), 1e-6)).asType(
                    x.dtype)
            }

            let q = normalize(parts[0]) * scale
            let k = normalize(parts[1])
            let betaRaw = writeGate(x).asType(.float32)
            let delta = writeModulation(x).asType(.float32)
            let ratio = exp(softplus(-betaRaw) - softplus(-betaRaw - delta))
            let v = (parts[2].reshaped(shape).asType(.float32) * ratio[.ellipsis, .newAxis]).asType(
                x.dtype)
            let decay =
                (decayB(decayA(x)).asType(.float32).reshaped(shape)
                    + dtBias.asType(.float32).reshaped(heads, headDim))
                * exp(aLog.asType(.float32)).reshaped(heads, 1)
            let g = exp(lowerBound.map { $0 * sigmoid(decay) } ?? -softplus(decay))
            let beta = sigmoid(betaRaw).asType(x.dtype)
            let state =
                cache?[1] ?? MLXArray.zeros([batch, heads, headDim, headDim], dtype: .float32)
            let (recurred, updated) =
                if !training, headDim.isMultiple(of: 32), Device.defaultDevice().deviceType == .gpu
                {
                    gatedDeltaKernel(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
                } else {
                    gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
                }
            cache?[1] = updated
            cache?.advance(length)
            cache?.offset += length
            let normed = MLXFast.rmsNorm(
                recurred.asType(.float32), weight: outputNorm.weight.asType(.float32), eps: epsilon)
            let gated = normed * sigmoid(gate(x).asType(.float32).reshaped(shape))
            return output(gated.asType(x.dtype).reshaped(batch, length, width))
        }
    }

    final class LatentAttention: Module {
        @ModuleInfo(key: "q_a_proj") var queryA: Linear
        @ModuleInfo(key: "q_a_layernorm") var queryNorm: RMSNorm
        @ModuleInfo(key: "q_b_proj") var queryB: Linear
        @ModuleInfo(key: "kv_a_proj_with_mqa") var latentProjection: Linear
        @ModuleInfo(key: "kv_a_layernorm") var latentNorm: RMSNorm
        @ModuleInfo(key: "embed_q") var embedQuery: MultiLinear
        @ModuleInfo(key: "unembed_out") var unembedOutput: MultiLinear
        @ModuleInfo(key: "o_proj") var output: Linear
        @ModuleInfo(key: "g_proj") var gate: Linear?

        let heads: Int
        let keyDim: Int
        let extraDim: Int
        let valueDim: Int
        let rank: Int
        let scale: Float

        init(_ config: Raon1MiniConfiguration) {
            heads = config.heads
            keyDim = config.keyHeadDim
            extraDim = config.extraKeyDim
            valueDim = config.valueHeadDim
            rank = config.latentRank
            scale = pow(Float(keyDim + extraDim), -0.5)
            _queryA.wrappedValue = Linear(config.hiddenSize, config.queryRank, bias: false)
            _queryNorm.wrappedValue = RMSNorm(
                dimensions: config.queryRank, eps: config.latentNormEpsilon)
            _queryB.wrappedValue = Linear(
                config.queryRank, heads * (keyDim + extraDim), bias: false)
            _latentProjection.wrappedValue = Linear(config.hiddenSize, rank + extraDim, bias: false)
            _latentNorm.wrappedValue = RMSNorm(dimensions: rank, eps: config.latentNormEpsilon)
            _embedQuery.wrappedValue = MultiLinear(
                inputDimensions: keyDim, outputDimensions: rank, numHeads: heads)
            _unembedOutput.wrappedValue = MultiLinear(
                inputDimensions: rank, outputDimensions: valueDim, numHeads: heads)
            _output.wrappedValue = Linear(heads * valueDim, config.hiddenSize, bias: false)
            if config.latentOutputGate {
                _gate.wrappedValue = Linear(config.hiddenSize, heads * valueDim, bias: false)
            }
        }

        func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
            let (batch, length) = (x.dim(0), x.dim(1))
            let mask = createAttentionMask(h: x, cache: cache, returnArray: true)
            let query = queryB(queryNorm(queryA(x))).reshaped(
                batch, length, heads, keyDim + extraDim
            )
            .transposed(0, 2, 1, 3)
            let q = split(query, indices: [keyDim], axis: -1)
            let kv = split(latentProjection(x), indices: [rank], axis: -1)
            var latent = latentNorm(kv[0]).expandedDimensions(axis: 1)
            var extra = kv[1].expandedDimensions(axis: 1)
            if let cache { (latent, extra) = cache.update(keys: latent, values: extra) }
            // The extra key channels contribute scores directly and have no RoPE.
            var scores = matmul(q[1] * scale, extra.swappedAxes(-1, -2))
            if case .array(let mask) = mask {
                scores = MLX.where(mask, scores, MLXArray(-Float.infinity, dtype: scores.dtype))
            }
            let absorbed =
                length
                <= maxAbsorbedQueries(
                    kvLoraRank: rank, qkNopeHeadDim: keyDim,
                    vHeadDim: valueDim, cacheLength: latentLength(latent))
            let attended = MLXFast.scaledDotProductAttention(
                queries: absorbed ? embedQuery(q[0]) : q[0],
                keys: absorbed ? latent : embedQuery(latent, transpose: false),
                values: absorbed ? latent : unembedOutput(latent),
                scale: scale, mask: .array(scores))
            var hidden = (absorbed ? unembedOutput(attended) : attended).transposed(0, 2, 1, 3)
                .reshaped(batch, length, heads * valueDim)
            if let gate { hidden = hidden * sigmoid(gate(x)) }
            return output(hidden)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        let activate: @Sendable (MLXArray, MLXArray) -> MLXArray

        init(_ config: Raon1MiniConfiguration, width: Int) {
            _gate.wrappedValue = Linear(config.hiddenSize, width, bias: false)
            _up.wrappedValue = Linear(config.hiddenSize, width, bias: false)
            _down.wrappedValue = Linear(width, config.hiddenSize, bias: false)
            activate = activation(config)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray { down(activate(gate(x), up(x))) }
    }

    final class Router: Module {
        let weight: MLXArray
        @ParameterInfo(key: "e_score_correction_bias") var bias: MLXArray
        let topK: Int
        let renormalize: Bool
        let scale: Float

        init(_ config: Raon1MiniConfiguration) {
            weight = .zeros([config.experts, config.hiddenSize])
            _bias.wrappedValue = .zeros([config.experts])
            topK = config.expertsPerToken
            renormalize = config.renormalize
            scale = config.expertScale
        }

        func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
            let scores = sigmoid(matmul(x.asType(.float32), weight.asType(.float32).T))
            let indices = stopGradient(
                argPartition(-(scores + bias.asType(.float32)), kth: topK - 1, axis: -1)[
                    .ellipsis, ..<topK])
            var weights = takeAlong(scores, indices, axis: -1)
            if renormalize { weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20) }
            return (indices, weights * scale)
        }
    }

    final class MoE: Module {
        let gate: Router
        @ModuleInfo(key: "switch_mlp") var experts: SwitchGLU
        @ModuleInfo(key: "shared_experts") var shared: MLP?

        init(_ config: Raon1MiniConfiguration) {
            gate = Router(config)
            _experts.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize, hiddenDims: config.expertWidth,
                numExperts: config.experts, activationProduct: activation(config))
            if config.sharedExperts > 0 {
                _shared.wrappedValue = MLP(config, width: config.expertWidth * config.sharedExperts)
            }
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (indices, weights) = gate(x)
            var result = (experts(x, indices).asType(.float32) * weights[.ellipsis, .newAxis]).sum(
                axis: -2)
            if let shared { result = result + shared(x).asType(.float32) }
            return result.asType(x.dtype)
        }
    }

    final class Layer: Module {
        @ModuleInfo(key: "self_attn") var attention: Module
        @ModuleInfo(key: "self_attn_raon_mhc") var attentionMixer: StreamMixer
        @ModuleInfo(key: "mlp_raon_mhc") var mlpMixer: StreamMixer
        @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
        let mlp: MLP?
        let moe: MoE?
        let isRecurrent: Bool

        init(_ config: Raon1MiniConfiguration, index: Int) {
            isRecurrent = config.delta.recurrentLayers.contains(index + 1)
            _attention.wrappedValue =
                isRecurrent ? DeltaAttention(config) as Module : LatentAttention(config)
            _attentionMixer.wrappedValue = StreamMixer(config)
            _mlpMixer.wrappedValue = StreamMixer(config)
            _inputNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEpsilon)
            _postAttentionNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEpsilon)
            if config.usesExperts(at: index) {
                moe = MoE(config)
                mlp = nil
            } else {
                mlp = MLP(config, width: config.intermediateSize)
                moe = nil
            }
        }

        func callAsFunction(_ streams: MLXArray, chart: Chart, cache: KVCache?) -> MLXArray {
            let (batch, length, count, width) = (
                streams.dim(0), streams.dim(1), streams.dim(2), streams.dim(3)
            )
            let rows = streams.reshaped(-1, count, width)
            let attentionRoutes = attentionMixer.routes(rows, chart: chart)
            let input = inputNorm(StreamMixer.aggregate(rows, pre: attentionRoutes.pre)).reshaped(
                batch, length, width)
            let attended =
                if let recurrent = attention as? DeltaAttention {
                    recurrent(input, cache: cache as? ArraysCache)
                } else { (attention as! LatentAttention)(input, cache: cache) }
            let mixed = StreamMixer.mix(
                rows, branch: attended.reshaped(-1, width), post: attentionRoutes.post,
                residual: attentionRoutes.residual)
            let routes = mlpMixer.routes(mixed, chart: chart)
            let normalized = postAttentionNorm(StreamMixer.aggregate(mixed, pre: routes.pre))
            let branch = moe.map { $0(normalized) } ?? mlp!(normalized)
            return StreamMixer.mix(
                mixed, branch: branch, post: routes.post, residual: routes.residual
            ).reshaped(streams.shape)
        }
    }

    final class Model: Module {
        @ModuleInfo(key: "embed_tokens") var embeddings: Embedding
        let layers: [Layer]
        let norm: RMSNorm
        let chart: Chart
        let streams: Int

        init(_ config: Raon1MiniConfiguration) {
            _embeddings.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            layers = (0 ..< config.layerCount).map { Layer(config, index: $0) }
            norm = RMSNorm(dimensions: config.hiddenSize, eps: config.normEpsilon)
            chart = Chart(config.streams)
            streams = config.streams
        }

        func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
            let h = embeddings(tokens)
            var residual = broadcast(
                h[0..., 0..., .newAxis, 0...], to: [h.dim(0), h.dim(1), streams, h.dim(2)])
            for (index, layer) in layers.enumerated() {
                residual = layer(residual, chart: chart, cache: cache?[index])
            }
            return norm(residual.asType(.float32).mean(axis: 2).asType(h.dtype))
        }
    }
}

public final class Raon1Mini: Module, LLMModel, UnwrappedLanguageModelWeights {
    let config: Raon1MiniConfiguration
    private let model: RaonLayers.Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: Raon1MiniConfiguration) {
        self.config = config
        model = RaonLayers.Model(config)
        if !config.tiedEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public var vocabularySize: Int { config.vocabularySize }
    public var loraLayers: [Module] { model.layers }
    public var toolCallFormat: ToolCallFormat? { .qwen35 }
    public var reasoningConfig: ReasoningConfig? { .thinkTagsWithEnableThinking }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        return lmHead.map { $0(hidden) } ?? model.embeddings.asLinear(hidden)
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        if let configuration, configuration.strategy.identifier != .fullPrecision {
            throw ModelExecutionError.invalidOptions(
                "Raon-1 Mini requires full-precision latent KV caches")
        }
        return try model.layers.map { layer in
            if layer.isRecurrent { return ArraysCache(size: 2) }
            return try makeAttentionKVCache(configuration: configuration)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = filterLMHeadWeights(
            from: Self.unwrapLanguageModelWeights(weights),
            tiedWordEmbeddings: config.tiedEmbeddings)
        if config.extraPredictionLayers > 0 {
            let end = config.layerCount + config.extraPredictionLayers
            for index in config.layerCount ..< end {
                let prefix = "model.layers.\(index)."
                for key in weights.keys where key.hasPrefix(prefix) {
                    weights.removeValue(forKey: key)
                }
            }
        }

        for (name, scales) in weights where name.hasSuffix(".weight_scale_inv") {
            let prefix = String(name.dropLast("weight_scale_inv".count))
            let path = prefix + "weight"
            guard let bytes = weights[path],
                bytes.ndim == 2, bytes.dtype == .uint8, scales.dtype == .uint8,
                bytes.dim(1).isMultiple(of: 32),
                scales.shape == [bytes.dim(0), bytes.dim(1) / 32],
                weights[prefix + "scales"] == nil
            else { continue }
            weights[path] = contiguous(bytes).view(dtype: .uint32)
            weights[prefix + "scales"] = scales
            weights.removeValue(forKey: name)
        }

        for (index, layer) in model.layers.enumerated() {
            let prefix = "model.layers.\(index)."
            if layer.isRecurrent {
                mergeConvolutions(weights: &weights, prefix: prefix + "self_attn.")
            } else {
                splitLatentProjection(weights: &weights, prefix: prefix + "self_attn.")
            }
            if layer.moe != nil {
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    for suffix in ["weight", "scales", "biases"] {
                        let target = prefix + "moe.switch_mlp.\(projection).\(suffix)"
                        let sources = (0 ..< config.experts).map {
                            prefix + "moe.experts.\($0).\(projection).\(suffix)"
                        }
                        let values = sources.compactMap { weights[$0] }
                        guard weights[target] == nil, values.count == config.experts else {
                            continue
                        }
                        weights[target] = stacked(values)
                        for key in sources { weights.removeValue(forKey: key) }
                    }
                }
            }
        }
        return weights
    }

    private func mergeConvolutions(weights: inout [String: MLXArray], prefix: String) {
        let keys = ["q", "k", "v"].map { prefix + "\($0)_conv1d.weight" }
        let combined = prefix + "qkv_conv1d.weight"
        func channelsLast(_ value: MLXArray) -> MLXArray {
            if value.ndim == 2 { return value.expandedDimensions(axis: -1) }
            if value.ndim == 3, value.dim(1) == 1, value.dim(2) != 1 {
                return value.swappedAxes(1, 2)
            }
            return value
        }
        if let fused = weights[combined] {
            weights[combined] = channelsLast(fused)
        } else {
            let values = keys.compactMap { weights[$0] }.map(channelsLast)
            let width = config.recurrentHeads * config.recurrentHeadDim
            guard values.count == 3,
                values.allSatisfy({ $0.shape == [width, config.delta.kernelSize, 1] })
            else { return }
            weights[combined] = concatenated(values, axis: 0)
            for key in keys { weights.removeValue(forKey: key) }
        }
    }

    private func splitLatentProjection(weights: inout [String: MLXArray], prefix: String) {
        let combined = prefix + "kv_b_proj.weight"
        let query = prefix + "embed_q.weight"
        let output = prefix + "unembed_out.weight"
        let dim = config.keyHeadDim + config.valueHeadDim
        guard let value = weights[combined],
            value.shape == [config.heads * dim, config.latentRank],
            weights[query] == nil, weights[output] == nil
        else { return }
        let parts = split(
            value.reshaped(config.heads, dim, config.latentRank), indices: [config.keyHeadDim],
            axis: 1)
        weights[query] = parts[0].swappedAxes(-1, -2)
        weights[output] = parts[1]
        weights.removeValue(forKey: combined)
    }

    func checkpointQuantization(_ original: BaseConfiguration.PerLayerQuantization?) throws
        -> BaseConfiguration.PerLayerQuantization?
    {
        if let original { return try Self.unwrapLanguageModelQuantization(original) }
        guard config.usesSourceMXFP8 else { return nil }
        return .init(
            quantization: .init(groupSize: 32, bits: 8, mode: .mxfp8), perLayerQuantization: [:])
    }
}
