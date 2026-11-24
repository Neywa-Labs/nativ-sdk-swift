import Foundation
import MLX
import MLXNN

enum Inkling2Language {
    typealias Configuration = Inkling2Configuration.Text

    final class Convolution: Module {
        @ModuleInfo(key: "conv") var convolution: Conv1d?
        @ModuleInfo(key: "weight_proj") var projection: Linear?
        @ParameterInfo(key: "static_weight") var staticWeight: MLXArray?
        let kernel: Int
        let index: Int
        let dynamicHeads: Int

        init(_ config: Configuration, channels: Int, index: Int, dynamicHeads: Int = 0) {
            kernel = config.kernel
            self.index = index
            self.dynamicHeads = dynamicHeads
            if dynamicHeads == 0 {
                _convolution.wrappedValue = Conv1d(
                    inputChannels: channels, outputChannels: channels, kernelSize: kernel,
                    groups: channels, bias: false)
            } else {
                _projection.wrappedValue = Linear(config.width, dynamicHeads * kernel, bias: false)
                _staticWeight.wrappedValue = .zeros([kernel, channels])
            }
        }

        func callAsFunction(
            _ x: MLXArray, conditionedOn conditioning: MLXArray? = nil,
            cache: ArraysCache?, mask: MLXArray?
        ) -> MLXArray {
            let (batch, length, channels) = (x.dim(0), x.dim(1), x.dim(2))
            let residual = x.asType(.float32)
            let inputs = mask.map { MLX.where($0[.ellipsis, .newAxis], residual, 0) } ?? residual
            let state = cache?[index] ?? MLXArray.zeros([batch, kernel - 1, channels])
            let context = concatenated([state, inputs], axis: 1)
            cache?[index] = nextInklingConvolutionState(state: state, inputs: inputs, mask: mask)

            if let convolution {
                let out = conv1d(context, convolution.weight.asType(.float32), groups: channels)
                return ((out + residual) / sqrt(Float(kernel + 1))).asType(x.dtype)
            }

            let dynamic = projection!(conditioning!).asType(.float32)
                .reshaped(batch, length, dynamicHeads, 1, kernel)
            let weights =
                staticWeight!.asType(.float32).transposed()
                .reshaped(dynamicHeads, channels / dynamicHeads, kernel) + dynamic
            let shape = [batch, length, dynamicHeads, channels / dynamicHeads]
            var out = MLXArray.zeros(shape)
            for lag in 0 ..< kernel {
                let start = kernel - 1 - lag
                let values = context[0..., start ..< (start + length), 0...].reshaped(shape)
                out = out + values * weights[.ellipsis, lag]
            }
            return ((out.reshaped(x.shape) + residual) / sqrt(Float(2 * kernel + 1)))
                .asType(x.dtype)
        }
    }

    private static func l2Clip(_ x: MLXArray, threshold: Float, scale: MLXArray? = nil)
        -> MLXArray
    {
        var value = x.asType(.float32)
        if let scale { value = value * scale.asType(.float32) }
        let norm = sqrt(square(value).sum(axis: -1, keepDims: true))
        return (value * minimum(threshold / (norm + 1e-6), 1)).asType(x.dtype)
    }

    final class Attention: Module {
        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "r_proj") var rProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "sink_proj") var sinkProj: Linear
        @ModuleInfo(key: "k_sconv") var keyConvolution: Convolution
        @ModuleInfo(key: "v_sconv") var valueConvolution: Convolution
        @ParameterInfo(key: "q_scale") var qScale: MLXArray
        @ParameterInfo(key: "rel_scale") var relativeScale: MLXArray
        @ParameterInfo(key: "rel_proj") var relativeProjection: MLXArray

        let config: Configuration
        let heads: Int
        let kvHeads: Int
        let headDimension: Int
        let sliding: Int
        let extent: Int

        init(_ config: Configuration, sliding: Bool) {
            self.config = config
            heads = sliding ? config.slidingHeads : config.heads
            kvHeads = sliding ? config.slidingKVHeads : config.kvHeads
            headDimension = sliding ? config.slidingHeadDimension : config.headDimension
            self.sliding = sliding ? config.window : 0
            extent = sliding ? config.window : config.relativeExtent
            let queryWidth = heads * headDimension
            let kvWidth = kvHeads * headDimension
            _qProj.wrappedValue = Linear(config.width, queryWidth, bias: false)
            _kProj.wrappedValue = Linear(config.width, kvWidth, bias: false)
            _vProj.wrappedValue = Linear(config.width, kvWidth, bias: false)
            _rProj.wrappedValue = Linear(
                config.width, heads * config.relativeDimension, bias: false)
            _oProj.wrappedValue = Linear(queryWidth, config.width, bias: false)
            _gateProj.wrappedValue = Linear(config.width, queryWidth, bias: false)
            _sinkProj.wrappedValue = Linear(config.width, heads, bias: false)
            _keyConvolution.wrappedValue = Convolution(
                config, channels: kvWidth, index: 0, dynamicHeads: sliding ? 0 : kvHeads)
            _valueConvolution.wrappedValue = Convolution(
                config, channels: kvWidth, index: 1, dynamicHeads: sliding ? 0 : kvHeads)
            _qScale.wrappedValue = .ones([heads, headDimension])
            _relativeScale.wrappedValue = .ones([heads, config.relativeDimension])
            _relativeProjection.wrappedValue = .zeros([config.relativeDimension, extent])
        }

        func callAsFunction(
            _ x: MLXArray, keyInput: MLXArray, valueInput: MLXArray,
            cache: KVCache?, convolutionCache: ArraysCache?, convolutionMask: MLXArray?
        ) -> MLXArray {
            let (batch, length) = (x.dim(0), x.dim(1))
            var q = l2Clip(
                qProj(x).reshaped(batch, length, heads, headDimension),
                threshold: config.qkClip, scale: qScale)
            var k = keyConvolution(
                kProj(keyInput), conditionedOn: keyInput,
                cache: convolutionCache, mask: convolutionMask)
            k = l2Clip(
                k.reshaped(batch, length, kvHeads, headDimension), threshold: config.qkClip)
            var v = valueConvolution(
                vProj(valueInput), conditionedOn: valueInput,
                cache: convolutionCache, mask: convolutionMask
            ).reshaped(batch, length, kvHeads, headDimension)
            if config.normalizeValues {
                v = MLXFast.rmsNorm(v.asType(.float32), weight: .mlxNone, eps: config.epsilon)
                    .asType(x.dtype)
            }
            var relative = l2Clip(
                rProj(x).reshaped(batch, length, heads, config.relativeDimension),
                threshold: config.relativeClip, scale: relativeScale)

            let offset = cache?.offset ?? 0
            if sliding == 0, let floor = config.logScalingFloor, offset + length > floor {
                let positions = MLXArray((offset + 1) ... (offset + length)).asType(.float32)
                let scaling = 1 + config.logScalingAlpha * log(maximum(positions / Float(floor), 1))
                let factor = scaling.reshaped(1, length, 1, 1)
                q = (q.asType(.float32) * factor).asType(x.dtype)
                relative = (relative.asType(.float32) * factor).asType(x.dtype)
            }
            q = q.transposed(0, 2, 1, 3)
            k = k.transposed(0, 2, 1, 3)
            v = v.transposed(0, 2, 1, 3)
            if let cache {
                (k, v) = cache.update(keys: k, values: v)
                // Relative logits index backwards from each query in temporal order.
                let storage = (cache as? MaterializedKVCache)?.storage ?? cache
                if let ring = storage as? RotatingKVCache,
                    let ordered = ring.logicalView(tail: sliding + length - 1)
                {
                    (k, v) = ordered
                }
            }

            let keyLength = k.dim(2)
            let leftPadding = convolutionCache?.leftPadding.map { $0 + (keyLength - length) }
            let bias = bandedAdditiveMask(
                rel: relative, projection: relativeProjection.asType(x.dtype),
                queryOffset: keyLength - length, keyLength: keyLength,
                slidingWindow: sliding, relativeExtent: extent, leftPadding: leftPadding,
                shapeReference: k)
            let sink = sinkProj(x).transposed(0, 2, 1).expandedDimensions(axis: -1)
            let zero = MLXArray.zeros([batch, kvHeads, 1, headDimension], dtype: x.dtype)
            let out = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: concatenated([k, zero], axis: 2),
                values: concatenated([v, zero], axis: 2),
                scale: 1 / sqrt(Float(headDimension)),
                mask: .array(concatenated([bias, sink], axis: -1))
            ).transposed(0, 2, 1, 3).reshaped(batch, length, heads * headDimension)
            return oProj(
                (out.asType(.float32) * sigmoid(gateProj(x).asType(.float32))).asType(x.dtype))
        }
    }

    final class Experts: Module {
        @ParameterInfo(key: "gate_weight") var router: MLXArray
        @ParameterInfo(key: "e_score_correction_bias") var correction: MLXArray
        @ModuleInfo(key: "switch_mlp") var projections: SwitchGLU

        let topK: Int
        let routeScale: Float

        init(_ config: Configuration) {
            topK = config.topK
            routeScale = config.routeScale
            _router.wrappedValue = .zeros([config.experts, config.width])
            _correction.wrappedValue = .zeros([config.experts])
            let gateCap = config.gateCap
            let upCap = config.upCap
            _projections.wrappedValue = SwitchGLU(
                inputDims: config.width, hiddenDims: config.expertWidth, numExperts: config.experts,
                activationProduct: { gate, up in
                    let g = gate.asType(.float32)
                    let u = up.asType(.float32)
                    return (tanh(g / gateCap) * gateCap * sigmoid(g) * tanh(u / upCap) * upCap)
                        .asType(gate.dtype)
                })
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let scores = sigmoid(matmul(x.asType(.float32), router.asType(.float32).T))
            let indices = argPartition(-(scores + correction), kth: topK - 1, axis: -1)[
                .ellipsis, ..<topK]
            let selected = takeAlong(scores, indices, axis: -1)
            let weights = selected / (selected.sum(axis: -1, keepDims: true) + 1e-20) * routeScale
            let out = projections(x, indices)
            return (out.asType(.float32) * weights[.ellipsis, .newAxis]).sum(axis: -2)
                .asType(out.dtype)
        }
    }

    final class AttentionResidual: Module {
        @ParameterInfo var query: MLXArray
        let normalizeValues: Bool

        init(_ config: Configuration, queries: Int = 1) {
            normalizeValues = config.normalizeResidualValues
            _query.wrappedValue = .zeros(queries == 1 ? [config.width] : [queries, config.width])
        }

        func callAsFunction(_ blocks: [MLXArray], partial: MLXArray) -> MLXArray {
            let width = partial.dim(-1)
            let states = stacked(blocks + [partial], axis: -2).asType(.float32)
            let keys = MLXFast.rmsNorm(states, weight: .mlxNone, eps: 1e-6)
            let scores =
                matmul(keys, query.asType(.float32).reshaped(-1, width).T)
                * (1 / sqrt(Float(width)))
            let weights = softmax(scores, axis: -2).swappedAxes(-1, -2)
            return matmul(weights, normalizeValues ? keys : states)
        }
    }

    final class Layer: Module {
        @ModuleInfo(key: "attention_input") var attentionInput: AttentionResidual?
        @ModuleInfo(key: "mlp_input") var mlpInput: AttentionResidual
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm?
        @ModuleInfo(key: "q_input_layernorm") var queryNorm: RMSNorm?
        @ModuleInfo(key: "k_input_layernorm") var keyNorm: RMSNorm?
        @ModuleInfo(key: "v_input_layernorm") var valueNorm: RMSNorm?
        @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
        @ModuleInfo(key: "attn_sconv") var attentionConvolution: Convolution
        @ModuleInfo(key: "mlp_sconv") var mlpConvolution: Convolution
        @ModuleInfo var mlp: Experts
        let startsBlock: Bool

        init(_ config: Configuration, index: Int) {
            let sliding = config.isSliding(index)
            startsBlock = index.isMultiple(of: config.blockSize)
            if index > 0 {
                _attentionInput.wrappedValue = AttentionResidual(config, queries: sliding ? 1 : 3)
            }
            _mlpInput.wrappedValue = AttentionResidual(config)
            _attention.wrappedValue = Attention(config, sliding: sliding)
            if sliding {
                _inputNorm.wrappedValue = RMSNorm(dimensions: config.width, eps: config.epsilon)
            } else {
                _queryNorm.wrappedValue = RMSNorm(dimensions: config.width, eps: config.epsilon)
                _keyNorm.wrappedValue = RMSNorm(dimensions: config.width, eps: config.epsilon)
                _valueNorm.wrappedValue = RMSNorm(dimensions: config.width, eps: config.epsilon)
            }
            _postNorm.wrappedValue = RMSNorm(dimensions: config.width, eps: config.epsilon)
            _attentionConvolution.wrappedValue = Convolution(
                config, channels: config.width, index: 2)
            _mlpConvolution.wrappedValue = Convolution(config, channels: config.width, index: 3)
            _mlp.wrappedValue = Experts(config)
        }

        func callAsFunction(_ x: MLXArray, blocks: inout [MLXArray], cache: CacheList?)
            -> MLXArray
        {
            let convolutions = cache?[1] as? ArraysCache
            let mask = inklingConvolutionPaddingMask(cache: convolutions, length: x.dim(1))
            let mixed = attentionInput?(blocks, partial: x) ?? x.expandedDimensions(axis: -2)
            let first = mixed[.ellipsis, 0, 0...]
            let q: MLXArray
            let k: MLXArray
            let v: MLXArray
            if let inputNorm {
                q = inputNorm(first).asType(x.dtype)
                k = q
                v = q
            } else {
                q = queryNorm!(first).asType(x.dtype)
                k = keyNorm!(mixed[.ellipsis, min(1, mixed.dim(-2) - 1), 0...]).asType(x.dtype)
                v = valueNorm!(mixed[.ellipsis, min(2, mixed.dim(-2) - 1), 0...]).asType(x.dtype)
            }
            let attended = attention(
                q, keyInput: k, valueInput: v, cache: cache?[0],
                convolutionCache: convolutions, convolutionMask: mask)
            var residual = attentionConvolution(attended, cache: convolutions, mask: mask)
                .asType(.float32)
            if startsBlock {
                blocks.append(x)
            } else {
                residual = residual + x.asType(.float32)
            }
            let hidden = mlpInput(blocks, partial: residual)[.ellipsis, 0, 0...]
            let routed = mlp(postNorm(hidden).asType(x.dtype))
            let projected = mlpConvolution(routed, cache: convolutions, mask: mask)
            convolutions?.advance(x.dim(1))
            return (residual + projected.asType(.float32)).asType(x.dtype)
        }
    }

    final class Backbone: Module {
        @ModuleInfo(key: "embed_tokens") var embedding: Embedding
        @ModuleInfo(key: "embed_norm") var embedNorm: RMSNorm
        @ModuleInfo var layers: [Layer]
        @ModuleInfo(key: "final_input") var finalInput: AttentionResidual
        @ModuleInfo var norm: RMSNorm

        init(_ config: Configuration) {
            _embedding.wrappedValue = Embedding(
                embeddingCount: config.vocabulary, dimensions: config.width)
            _embedNorm.wrappedValue = RMSNorm(dimensions: config.width, eps: config.epsilon)
            _layers.wrappedValue = (0 ..< config.layers).map { Layer(config, index: $0) }
            _finalInput.wrappedValue = AttentionResidual(config)
            _norm.wrappedValue = RMSNorm(dimensions: config.width, eps: config.epsilon)
        }

        func embed(_ tokens: MLXArray) -> MLXArray {
            embedNorm(embedding(tokens))
        }

        func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]?, lastOnly: Bool) -> MLXArray {
            var x = embeddings
            var blocks: [MLXArray] = []
            for (index, layer) in layers.enumerated() {
                x = layer(x, blocks: &blocks, cache: cache?[index] as? CacheList)
            }
            if lastOnly && x.dim(1) > 1 {
                x = x[0..., (-1)..., 0...]
                blocks = blocks.map { $0[0..., (-1)..., 0...] }
            }
            return norm(finalInput(blocks, partial: x)[.ellipsis, 0, 0...])
        }
    }

    final class Model: Module {
        @ModuleInfo var model: Backbone
        @ModuleInfo(key: "lm_head") var head: Linear
        let config: Configuration

        init(_ config: Configuration) {
            self.config = config
            _model.wrappedValue = Backbone(config)
            _head.wrappedValue = Linear(config.width, config.vocabulary, bias: false)
        }

        func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]?, lastOnly: Bool = false)
            -> MLXArray
        {
            let normalized = model(embeddings, cache: cache, lastOnly: lastOnly)
                .asType(model.embedNorm.weight.dtype).asType(.float32)
            // The original mixed-precision checkpoint stores a BF16 head with FP32 logits.
            var logits = head(normalized)
            if config.logitCap > 0 {
                logits = tanh(logits / config.logitCap) * config.logitCap
            }
            return logits[.ellipsis, ..<(config.unpaddedVocabulary ?? config.vocabulary)]
        }
    }
}
