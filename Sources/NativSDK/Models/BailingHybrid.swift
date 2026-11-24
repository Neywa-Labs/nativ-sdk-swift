// Architecture adapted from https://huggingface.co/inclusionAI/Ling-3.1-flash (MIT).
import Foundation
import MLX
import MLXNN

enum BailingHybridLayers {
    private static let clippedSwiGLU = compile(shapeless: true) { gate, up, limit in
        minimum(silu(gate), limit) * minimum(maximum(up, -limit), limit)
    }

    static func activation(limit: Float?) -> @Sendable (MLXArray, MLXArray) -> MLXArray {
        guard let limit, limit > 0 else { return compiledSiluProduct }
        return { gate, up in clippedSwiGLU(gate, up, MLXArray(limit, dtype: gate.dtype)) }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        let activate: @Sendable (MLXArray, MLXArray) -> MLXArray

        init(_ config: BailingHybridConfiguration, width: Int, limit: Float? = nil) {
            _gate.wrappedValue = Linear(config.hiddenSize, width, bias: false)
            _up.wrappedValue = Linear(config.hiddenSize, width, bias: false)
            _down.wrappedValue = Linear(width, config.hiddenSize, bias: false)
            activate = activation(limit: limit)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray { down(activate(gate(x), up(x))) }
    }

    final class Router: Module {
        let weight: MLXArray
        @ParameterInfo(key: "expert_bias") var expertBias: MLXArray
        let config: BailingHybridConfiguration

        init(_ config: BailingHybridConfiguration) {
            self.config = config
            weight = .zeros([config.experts, config.hiddenSize], dtype: .float32)
            _expertBias.wrappedValue = .zeros([config.experts], dtype: .float32)
        }

        func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
            let logits = matmul(x.asType(.float32), weight.asType(.float32).T)
            return groupedExpertSelection(
                gates: logits, correctionBias: expertBias, topK: config.expertsPerToken,
                groups: config.groupsPerToken < config.expertGroups ? config.expertGroups : 1,
                topGroups: config.groupsPerToken, routedScalingFactor: config.expertScale,
                normalizeTopK: config.normalizeExperts)
        }
    }

    final class MoE: Module, UnaryLayer {
        let gate: Router
        @ModuleInfo(key: "switch_mlp") var experts: SwitchGLU
        @ModuleInfo(key: "shared_experts") var shared: MLP?

        init(_ config: BailingHybridConfiguration, index: Int) {
            gate = Router(config)
            let expertClip =
                config.expertClips.indices.contains(index) ? config.expertClips[index] : nil
            _experts.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize, hiddenDims: config.expertWidth,
                numExperts: config.experts, activationProduct: activation(limit: expertClip))
            if config.sharedExperts > 0 {
                let sharedClip =
                    config.sharedClips.indices.contains(index) ? config.sharedClips[index] : nil
                _shared.wrappedValue = MLP(
                    config, width: config.sharedWidth * config.sharedExperts, limit: sharedClip)
            }
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (indices, weights) = gate(x)
            let routed = (experts(x, indices).asType(.float32) * weights[.ellipsis, .newAxis])
                .sum(axis: -2).asType(x.dtype)
            return shared.map { routed + $0(x) } ?? routed
        }
    }

    final class DeltaAttention: Module {
        @ModuleInfo(key: "q_proj") var query: Linear
        @ModuleInfo(key: "k_proj") var key: Linear
        @ModuleInfo(key: "v_proj") var value: Linear
        @ModuleInfo(key: "q_conv1d") var queryConv: KimiK3Text.ShortConv1d
        @ModuleInfo(key: "k_conv1d") var keyConv: KimiK3Text.ShortConv1d
        @ModuleInfo(key: "v_conv1d") var valueConv: KimiK3Text.ShortConv1d
        @ModuleInfo(key: "f_proj") var decay: Linear?
        @ModuleInfo(key: "f_a_proj") var decayA: Linear?
        @ModuleInfo(key: "f_b_proj") var decayB: Linear?
        @ModuleInfo(key: "g_proj") var outputGate: Linear?
        @ModuleInfo(key: "g_a_proj") var outputGateA: Linear?
        @ModuleInfo(key: "g_b_proj") var outputGateB: Linear?
        @ModuleInfo(key: "b_proj") var writeGate: Linear
        @ModuleInfo(key: "o_norm") var outputNorm: RMSNorm
        @ModuleInfo(key: "o_proj") var output: Linear
        @ParameterInfo(key: "A_log") var aLog: MLXArray
        @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
        let heads: Int
        let headDim: Int
        let scale: Float
        let lowerBound: Float?

        init(_ config: BailingHybridConfiguration) {
            heads = config.heads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)
            lowerBound = config.lowerBound
            let width = heads * headDim
            let hidden = config.hiddenSize
            _query.wrappedValue = Linear(hidden, width, bias: false)
            _key.wrappedValue = Linear(hidden, width, bias: false)
            _value.wrappedValue = Linear(hidden, width, bias: false)
            _queryConv.wrappedValue = .init(
                channels: width, kernelSize: config.convolutionSize, fuseDecode: true)
            _keyConv.wrappedValue = .init(
                channels: width, kernelSize: config.convolutionSize, fuseDecode: true)
            _valueConv.wrappedValue = .init(
                channels: width, kernelSize: config.convolutionSize, fuseDecode: true)
            _writeGate.wrappedValue = Linear(hidden, heads, bias: false)
            if config.fullRankKDA {
                _decay.wrappedValue = Linear(hidden, width, bias: false)
                _outputGate.wrappedValue = Linear(hidden, width, bias: false)
            } else {
                _decayA.wrappedValue = Linear(hidden, headDim, bias: false)
                _decayB.wrappedValue = Linear(headDim, width, bias: false)
                _outputGateA.wrappedValue = Linear(hidden, headDim, bias: false)
                _outputGateB.wrappedValue = Linear(headDim, width, bias: false)
            }
            _outputNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEpsilon)
            _output.wrappedValue = Linear(width, hidden, bias: false)
            _aLog.wrappedValue = .zeros([heads], dtype: .float32)
            _dtBias.wrappedValue = .zeros([width], dtype: .float32)
        }

        func callAsFunction(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
            let (batch, length) = (x.dim(0), x.dim(1))
            let shape = [batch, length, heads, headDim]
            let lengths = cache?.currentLengths
            let mask = cache?.makeMask(N: length)
            let (q, qState) = queryConv(query(x), state: cache?[0], mask: mask, lengths: lengths)
            let (k, kState) = keyConv(key(x), state: cache?[1], mask: mask, lengths: lengths)
            let (v, vState) = valueConv(value(x), state: cache?[2], mask: mask, lengths: lengths)
            cache?[0] = qState
            cache?[1] = kState
            cache?[2] = vState
            let eps = Float(1e-6) / Float(headDim)
            let queries =
                (1 / Float(headDim))
                * MLXFast.rmsNorm(
                    q.reshaped(shape), weight: .mlxNone, eps: eps)
            let keys = scale * MLXFast.rmsNorm(k.reshaped(shape), weight: .mlxNone, eps: eps)
            let values = v.reshaped(shape)

            let a = (decay.map { $0(x) } ?? decayB!(decayA!(x))).asType(.float32).reshaped(shape)
            let bias = dtBias.asType(.float32).reshaped(heads, headDim)
            let aLog = aLog.asType(.float32).reshaped(heads, 1)
            let g =
                lowerBound.map { exp($0 * sigmoid(exp(aLog) * (a + bias))) }
                ?? computeGatedDeltaG(aLog, a, bias)
            let beta = sigmoid(writeGate(x).asType(.float32))
            let state =
                cache?[3] ?? MLXArray.zeros([batch, heads, headDim, headDim], dtype: .float32)
            let (out, updated) =
                if !training, headDim.isMultiple(of: 32), Device.defaultDevice().deviceType == .gpu
                {
                    gatedDeltaKernel(
                        q: queries, k: keys, v: values, g: g, beta: beta, state: state, mask: mask)
                } else {
                    gatedDeltaOps(
                        q: queries, k: keys, v: values, g: g, beta: beta, state: state, mask: mask)
                }
            cache?[3] = updated
            cache?.advance(length)
            cache?.offset += length
            let gate = outputGate.map { $0(x) } ?? outputGateB!(outputGateA!(x))
            let normalized = outputNorm(out).reshaped(batch, length, -1)
            return output(normalized * sigmoid(gate.asType(.float32)).asType(x.dtype))
        }
    }

    final class LatentAttention: Module {
        @ModuleInfo(key: "q_proj") var query: Linear?
        @ModuleInfo(key: "q_a_proj") var queryA: Linear?
        @ModuleInfo(key: "q_a_layernorm") var queryNorm: RMSNorm?
        @ModuleInfo(key: "q_b_proj") var queryB: Linear?
        @ModuleInfo(key: "kv_a_proj_with_mqa") var latentProjection: Linear
        @ModuleInfo(key: "kv_a_layernorm") var latentNorm: RMSNorm
        @ModuleInfo(key: "embed_q") var embedQuery: MultiLinear
        @ModuleInfo(key: "unembed_out") var unembedOutput: MultiLinear
        @ModuleInfo(key: "g_proj") var gate: Linear?
        @ModuleInfo(key: "dense") var output: Linear
        let config: BailingHybridConfiguration
        let rope: RoPELayer
        let scale: Float

        init(_ config: BailingHybridConfiguration) {
            self.config = config
            let queryWidth = config.heads * (config.keyDim + config.ropeDim)
            if let queryRank = config.queryRank {
                _queryA.wrappedValue = Linear(config.hiddenSize, queryRank, bias: config.qkvBias)
                _queryNorm.wrappedValue = RMSNorm(dimensions: queryRank, eps: 1e-6)
                _queryB.wrappedValue = Linear(queryRank, queryWidth, bias: false)
            } else {
                _query.wrappedValue = Linear(config.hiddenSize, queryWidth, bias: false)
            }
            _latentProjection.wrappedValue = Linear(
                config.hiddenSize, config.latentRank + config.ropeDim, bias: config.qkvBias)
            _latentNorm.wrappedValue = RMSNorm(dimensions: config.latentRank, eps: 1e-6)
            _embedQuery.wrappedValue = MultiLinear(
                inputDimensions: config.keyDim, outputDimensions: config.latentRank,
                numHeads: config.heads)
            _unembedOutput.wrappedValue = MultiLinear(
                inputDimensions: config.latentRank, outputDimensions: config.valueDim,
                numHeads: config.heads)
            if let gate = config.attentionGate {
                let size = gate == .head ? config.heads : config.heads * config.valueDim
                _gate.wrappedValue = Linear(config.hiddenSize, size, bias: false)
            }
            _output.wrappedValue = Linear(
                config.heads * config.valueDim, config.hiddenSize, bias: config.qkvBias)
            rope = initializeRope(
                dims: config.ropeDim, base: config.ropeTheta, traditional: config.ropeInterleaved,
                scalingConfig: config.ropeScaling,
                maxPositionEmbeddings: config.maxPositionEmbeddings)
            let factor = config.ropeScaling?["factor"]?.asFloat() ?? 1
            let mscale = config.ropeScaling?["mscale_all_dim"]?.asFloat() ?? 0
            let correction = factor > 1 ? pow(1 + 0.1 * mscale * log(factor), 2) : 1
            scale = pow(Float(config.keyDim + config.ropeDim), -0.5) * correction
        }

        func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
            let (batch, length) = (x.dim(0), x.dim(1))
            let mask = createAttentionMask(h: x, cache: cache, returnArray: true)
            let projected = query.map { $0(x) } ?? queryB!(queryNorm!(queryA!(x)))
            let q = projected.reshaped(batch, length, config.heads, config.keyDim + config.ropeDim)
                .transposed(0, 2, 1, 3)
            let queryParts = split(q, indices: [config.keyDim], axis: -1)
            let keyParts = split(latentProjection(x), indices: [config.latentRank], axis: -1)
            let offset = cache?.ropeOffset
            let qPosition = applyRotaryPosition(rope, to: queryParts[1], offset: offset)
            var kPosition = applyRotaryPosition(
                rope, to: keyParts[1].expandedDimensions(axis: 1), offset: offset)
            var latent = latentNorm(keyParts[0]).expandedDimensions(axis: 1)
            if let cache { (latent, kPosition) = cache.update(keys: latent, values: kPosition) }
            var bias = matmul(qPosition * scale, kPosition.swappedAxes(-1, -2))
            if case .array(let mask) = mask {
                // Finite masked scores also keep fully padded query rows finite.
                let minimum =
                    bias.dtype == .float16
                    ? -Float(Float16.greatestFiniteMagnitude)
                    : (bias.dtype == .bfloat16
                        ? -Float(bitPattern: 0x7f7f_0000) : -Float.greatestFiniteMagnitude)
                bias = MLX.where(mask, bias, MLXArray(minimum, dtype: bias.dtype))
            }
            let absorbed = length == 1
            let attended = MLXFast.scaledDotProductAttention(
                queries: absorbed ? embedQuery(queryParts[0]) : queryParts[0],
                keys: absorbed ? latent : embedQuery(latent, transpose: false),
                values: absorbed ? latent : unembedOutput(latent),
                scale: scale, mask: .array(bias))
            var hidden = (absorbed ? unembedOutput(attended) : attended).transposed(0, 2, 1, 3)
            if let gate {
                hidden =
                    hidden
                    * sigmoid(gate(x).asType(.float32)).asType(x.dtype)
                    .reshaped(batch, length, config.heads, -1)
            }
            return output(hidden.reshaped(batch, length, -1))
        }
    }

    final class Layer: Module {
        @ModuleInfo var attention: Module
        @ModuleInfo var mlp: Module
        @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
        let isRecurrent: Bool

        init(_ config: BailingHybridConfiguration, index: Int) {
            isRecurrent = config.isRecurrent(index)
            _attention.wrappedValue =
                isRecurrent ? DeltaAttention(config) as Module : LatentAttention(config)
            _mlp.wrappedValue =
                config.usesExperts(index)
                ? MoE(config, index: index) as Module : MLP(config, width: config.intermediateSize)
            _inputNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEpsilon)
            _postAttentionNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEpsilon)
        }

        func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
            let input = inputNorm(x)
            let attended =
                if let delta = attention as? DeltaAttention {
                    delta(input, cache: cache as? ArraysCache)
                } else { (attention as! LatentAttention)(input, cache: cache) }
            let residual = x + attended
            return residual + (mlp as! UnaryLayer)(postAttentionNorm(residual))
        }
    }

    final class Model: Module {
        @ModuleInfo(key: "word_embeddings") var embeddings: Embedding
        let layers: [Layer]
        let norm: RMSNorm

        init(_ config: BailingHybridConfiguration) {
            _embeddings.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            layers = (0 ..< config.layerCount).map { Layer(config, index: $0) }
            norm = RMSNorm(dimensions: config.hiddenSize, eps: config.normEpsilon)
        }

        func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
            var hidden = embeddings(tokens)
            for (index, layer) in layers.enumerated() {
                hidden = layer(hidden, cache: cache?[index])
            }
            return norm(hidden)
        }
    }
}

public final class BailingHybrid: Module, LLMModel, UnwrappedLanguageModelWeights {
    let config: BailingHybridConfiguration
    private let model: BailingHybridLayers.Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: BailingHybridConfiguration) {
        self.config = config
        model = BailingHybridLayers.Model(config)
        if !config.tiedEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public var vocabularySize: Int { config.vocabularySize }
    public var loraLayers: [Module] { model.layers }
    public var toolCallFormat: ToolCallFormat? { .ling3 }
    public var reasoningConfig: ReasoningConfig? {
        .init(
            startDelimiter: "<think>", endDelimiter: "</think>",
            promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true),
            implicitEndDelimiters: ["<tool_call>"])
    }

    public func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        ObjectToolArgumentsMessageGenerator()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        return lmHead.map { $0(hidden) } ?? model.embeddings.asLinear(hidden)
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        if let configuration, configuration.strategy.identifier != .fullPrecision {
            throw ModelExecutionError.invalidOptions(
                "Ling requires full-precision latent KV caches")
        }
        return try model.layers.map { layer in
            if layer.isRecurrent { return ArraysCache(size: 4) }
            return try makeAttentionKVCache(configuration: configuration)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = filterLMHeadWeights(
            from: Self.unwrapLanguageModelWeights(weights),
            tiedWordEmbeddings: config.tiedEmbeddings)
        if config.predictionLayers > 0 {
            let sidecar = config.layerCount ..< config.layerCount + config.predictionLayers
            for key in weights.keys where key.hasPrefix("model.layers.") {
                let index = key.dropFirst("model.layers.".count).prefix(while: { $0 != "." })
                if let index = Int(index), sidecar.contains(index) {
                    weights.removeValue(forKey: key)
                }
            }
        }
        for (key, scales) in weights where key.hasSuffix(".weight_scale_inv") {
            let weight = String(key.dropLast("_scale_inv".count))
            guard let raw = weights[weight], raw.ndim == 2, raw.dtype == .uint8,
                scales.shape == [(raw.dim(0) + 127) / 128, (raw.dim(1) + 127) / 128]
            else { continue }
            weights[weight] = DeepseekV32Model.dequantizeBlockScaled(raw, scaleInverse: scales)
            weights.removeValue(forKey: key)
        }
        for index in 0 ..< config.layerCount {
            let prefix = "model.layers.\(index)."
            if config.usesExperts(index) {
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    for suffix in ["weight", "scales", "biases"] {
                        let target = prefix + "mlp.switch_mlp.\(projection).\(suffix)"
                        let sources = (0 ..< config.experts).map {
                            prefix + "mlp.experts.\($0).\(projection).\(suffix)"
                        }
                        let values = sources.compactMap { weights[$0] }
                        guard weights[target] == nil, values.count == config.experts else {
                            continue
                        }
                        weights[target] = stacked(values)
                        for source in sources { weights.removeValue(forKey: source) }
                    }
                }
            }
            let attention = prefix + "attention."
            if config.isRecurrent(index) {
                for name in ["q", "k", "v"] {
                    let source = attention + "\(name)_conv1d.weight"
                    let target = attention + "\(name)_conv1d.conv.weight"
                    guard let raw = weights[source], weights[target] == nil else { continue }
                    let conv =
                        raw.ndim == 2
                        ? raw.expandedDimensions(axis: -1)
                        : raw.ndim == 3 && raw.dim(1) == 1 ? raw.swappedAxes(1, 2) : raw
                    weights[target] = conv
                    weights.removeValue(forKey: source)
                }
            } else {
                splitLatentProjection(prefix: attention, weights: &weights)
            }
        }
        return weights
    }

    private func splitLatentProjection(prefix: String, weights: inout [String: MLXArray]) {
        let source = prefix + "kv_b_proj"
        let embed = prefix + "embed_q"
        let unembed = prefix + "unembed_out"
        let width = config.keyDim + config.valueDim
        let rows = config.heads * width
        let rank = config.latentRank
        guard var combined = weights[source + ".weight"],
            combined.ndim == 2, combined.dim(0) == rows,
            weights[embed + ".weight"] == nil, weights[unembed + ".weight"] == nil
        else { return }

        let quantization: BaseConfiguration.Quantization?
        if let scales = weights[source + ".scales"] {
            guard combined.dtype == .uint32, scales.ndim == 2, scales.dim(0) == rows,
                scales.dim(1) > 0, rank.isMultiple(of: scales.dim(1)),
                (combined.dim(1) * 32).isMultiple(of: rank)
            else { return }
            let bits = combined.dim(1) * 32 / rank
            let mode: QuantizationMode =
                scales.dtype == .uint8 ? (bits == 4 ? .mxfp4 : .mxfp8) : .affine
            let parameters = BaseConfiguration.Quantization(
                groupSize: rank / scales.dim(1), bits: bits, mode: mode)
            let biases = weights[source + ".biases"]
            guard (try? parameters.validateForConversion()) != nil,
                mode != .affine || biases?.shape == scales.shape
            else { return }
            combined = dequantized(
                combined, scales: scales, biases: biases, groupSize: parameters.groupSize,
                bits: parameters.bits, mode: parameters.mode)
            quantization = parameters
        } else {
            quantization = nil
        }
        guard combined.dim(1) == rank else { return }
        let parts = split(
            combined.reshaped(config.heads, width, rank), indices: [config.keyDim], axis: 1)
        for (path, projection) in [(embed, parts[0].swappedAxes(-1, -2)), (unembed, parts[1])] {
            let projection = contiguous(projection)
            if let quantization, projection.dim(-1).isMultiple(of: quantization.groupSize) {
                let packed = quantized(
                    projection, groupSize: quantization.groupSize, bits: quantization.bits,
                    mode: quantization.mode)
                weights[path + ".weight"] = packed.wq
                weights[path + ".scales"] = packed.scales
                weights[path + ".biases"] = packed.biases
            } else {
                weights[path + ".weight"] = projection
            }
        }
        for suffix in ["weight", "scales", "biases"] {
            weights.removeValue(forKey: source + "." + suffix)
        }
    }

    func checkpointQuantization(_ original: BaseConfiguration.PerLayerQuantization?) throws
        -> BaseConfiguration.PerLayerQuantization?
    {
        guard var quantization = try Self.unwrapLanguageModelQuantization(original) else {
            return nil
        }
        for index in 0 ..< config.layerCount where !config.isRecurrent(index) {
            let prefix = "model.layers.\(index).attention."
            if let option = quantization.perLayerQuantization[prefix + "kv_b_proj"] {
                for name in ["embed_q", "unembed_out"]
                where quantization.perLayerQuantization[prefix + name] == nil {
                    quantization.perLayerQuantization[prefix + name] = option
                }
            }
        }
        return quantization
    }

    public func quantizationPredicate(path: String, module: Module) -> QuantizationDecision {
        if [".mlp.gate", ".attention.b_proj", ".attention.g_proj"].contains(where: path.hasSuffix) {
            return .skip
        }
        return .useDefault
    }
}
