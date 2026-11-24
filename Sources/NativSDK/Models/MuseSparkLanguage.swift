import Foundation
import MLX
import MLXNN

// Apply SwiGLU in FP32 and round once before the expert down projection.
private let museSparkSwiGLU = compile(shapeless: true) { gate, up in
    (silu(gate.asType(.float32)) * up.asType(.float32)).asType(up.dtype)
}

enum MuseSparkLanguage {
    static func gateCoefficients(_ gate: MLXArray, temperature: Float) -> (MLXArray, MLXArray) {
        let logits = gate.asType(.float32) / temperature
        let beta = sigmoid(logits)
        return (sqrt(maximum(sigmoid(-logits) * (1 + beta), 1e-3)), beta)
    }

    private static func preNorm(_ x: MLXArray, _ norm: RMSNorm, dtype: DType) -> MLXArray {
        MLXFast.rmsNorm(x, weight: norm.weight.asType(.float32), eps: norm.eps).asType(dtype)
    }

    final class Attention: Module {
        @ModuleInfo(key: "qkvg_proj") var input: Linear
        @ModuleInfo(key: "o_proj") var output: Linear
        let config: MuseSparkConfiguration.Text
        let rope: RoPE?
        let scale: Float

        init(_ config: MuseSparkConfiguration.Text, usesRoPE: Bool) {
            self.config = config
            let qWidth = config.heads * config.headDimension
            let kvWidth = config.kvHeads * config.headDimension
            _input.wrappedValue = Linear(config.hiddenSize, 2 * (qWidth + kvWidth), bias: false)
            _output.wrappedValue = Linear(qWidth, config.hiddenSize, bias: false)
            rope =
                usesRoPE
                ? RoPE(dimensions: config.headDimension, traditional: true, base: config.ropeBase)
                : nil
            scale = pow(Float(config.headDimension), -0.5)
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let (batch, length, _) = x.shape3
            let dimension = config.headDimension
            let qWidth = config.heads * dimension
            let kvWidth = config.kvHeads * dimension
            let parts = split(
                input(x), indices: [qWidth, qWidth + kvWidth, qWidth + 2 * kvWidth], axis: -1)
            let queryShape = [batch, length, config.heads, dimension]
            let kvShape = [batch, length, config.kvHeads, dimension]
            let q = MLXFast.rmsNorm(
                parts[0].reshaped(queryShape), weight: .mlxNone, eps: config.rmsEpsilon)
            var queries = (q.asType(.float32) * config.queryScale).asType(q.dtype).transposed(
                0, 2, 1, 3)
            var keys = MLXFast.rmsNorm(
                parts[1].reshaped(kvShape), weight: .mlxNone, eps: config.rmsEpsilon
            )
            .transposed(0, 2, 1, 3)
            let values = parts[2].reshaped(kvShape).transposed(0, 2, 1, 3)
            if let rope {
                queries = applyRotaryPosition(rope, to: queries, offset: cache?.ropeOffset)
                keys = applyRotaryPosition(rope, to: keys, offset: cache?.ropeOffset)
            }
            let attended = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
            )
            let normalized = MLXFast.rmsNorm(
                attended.asType(.float32), weight: .mlxNone, eps: config.rmsEpsilon
            )
            .transposed(0, 2, 1, 3).reshaped(batch, length, qWidth)
            let gated = (normalized * sigmoid(parts[3].asType(.float32))).asType(attended.dtype)
            return output(gated)
        }
    }

    final class Router: Module {
        let weight: MLXArray
        @ParameterInfo(key: "e_score_correction_bias") var correction: MLXArray
        let config: MuseSparkConfiguration.Text

        init(_ config: MuseSparkConfiguration.Text) {
            self.config = config
            weight = .zeros([config.expertCount, config.hiddenSize], dtype: .float32)
            _correction.wrappedValue = .zeros([config.expertCount], dtype: .float32)
        }

        func callAsFunction(_ input: MLXArray) -> (MLXArray, MLXArray) {
            let logits =
                matmul(input.asType(.float32), weight.asType(.float32).T)
                * config.outputMultiplier
            let magnitude = 1 / (1 + exp(abs(logits)))
            let scores = MLX.where(logits .< 0, magnitude, 1 - magnitude)
            let order = argSort(-(scores + correction.asType(.float32)), axis: -1)
            let indices = order[.ellipsis, ..<config.topK]
            let selected = takeAlong(scores, indices, axis: -1)
            let scaled =
                selected / (selected.sum(axis: -1, keepDims: true) + 1e-15)
                * config.routeScale
            return (indices, scaled)
        }
    }

    final class InputScale: Module {
        let weight: MLXArray

        init(_ width: Int) { weight = .ones([width]) }
    }

    final class Experts: SwitchGLU {
        @ModuleInfo(key: "post_expert_norm") var post: InputScale

        init(_ config: MuseSparkConfiguration.Text) {
            _post.wrappedValue = InputScale(config.expertWidth)
            super.init(
                inputDims: config.expertWidth, hiddenDims: config.expertIntermediate,
                numExperts: config.expertCount, activationProduct: museSparkSwiGLU)
        }
    }

    final class MoE: Module {
        let gate: Router
        @ModuleInfo(key: "pre_expert_proj") var compress: Linear
        @ModuleInfo(key: "pre_expert_norm") var preNorm: RMSNorm
        let experts: Experts
        @ModuleInfo(key: "post_expert_proj") var expand: Linear
        let config: MuseSparkConfiguration.Text

        init(_ config: MuseSparkConfiguration.Text) {
            self.config = config
            gate = Router(config)
            _compress.wrappedValue = Linear(config.hiddenSize, config.expertWidth, bias: false)
            _preNorm.wrappedValue = RMSNorm(dimensions: config.expertWidth, eps: config.rmsEpsilon)
            experts = Experts(config)
            _expand.wrappedValue = Linear(config.expertWidth, config.hiddenSize, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (indices, scores) = gate(x)
            let computed = experts(preNorm(compress(x)), indices)
            let normalized = MLXFast.rmsNorm(
                computed.asType(.float32) * experts.post.weight.asType(.float32),
                weight: .mlxNone, eps: config.postEpsilon)
            let weights = scores.asType(computed.dtype).asType(.float32)
            var combined = MLXArray.zeros(
                Array(computed.shape.dropLast(2)) + [config.expertWidth], dtype: computed.dtype)
            // Each addition is rounded to the expert dtype before adding the next route.
            for index in 0 ..< config.topK {
                combined =
                    (combined.asType(.float32)
                    + normalized[.ellipsis, index, 0...] * weights[.ellipsis, index, .newAxis])
                    .asType(computed.dtype)
            }
            return expand(combined)
        }
    }

    final class Block: Module {
        @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "pre_feedforward_layernorm") var feedforwardNorm: RMSNorm
        @ModuleInfo(key: "post_feedforward_layernorm") var postScale: InputScale
        let mlp: MoE
        @ParameterInfo(key: "add_attn_alpha") var attentionAlpha: MLXArray
        @ParameterInfo(key: "add_attn_beta") var attentionBeta: MLXArray
        @ParameterInfo(key: "add_ffn_alpha") var feedforwardAlpha: MLXArray
        @ParameterInfo(key: "add_ffn_beta") var feedforwardBeta: MLXArray
        let epsilon: Float

        init(_ config: MuseSparkConfiguration.Text, index: Int) {
            epsilon = config.postEpsilon
            _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsEpsilon)
            _attention.wrappedValue = Attention(config, usesRoPE: config.layerTheta[index] != 0)
            _feedforwardNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsEpsilon)
            _postScale.wrappedValue = InputScale(config.hiddenSize)
            mlp = MoE(config)
            let (alpha, beta) = gateCoefficients(
                .zeros([config.hiddenSize], dtype: .float32), temperature: config.gateTemperature)
            _attentionAlpha.wrappedValue = alpha
            _attentionBeta.wrappedValue = beta
            _feedforwardAlpha.wrappedValue = alpha[.ellipsis]
            _feedforwardBeta.wrappedValue = beta[.ellipsis]
        }

        func callAsFunction(
            _ stream: MLXArray, dtype: DType, mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?
        ) -> (MLXArray, DType) {
            let attended = attention(
                preNorm(stream, inputNorm, dtype: dtype), mask: mask, cache: cache)
            let branch = MLXFast.rmsNorm(attended, weight: .mlxNone, eps: epsilon).asType(.float32)
            let residual =
                stream * attentionAlpha.asType(.float32)
                + branch * attentionBeta.asType(.float32)
            let expert = mlp(preNorm(residual, feedforwardNorm, dtype: attended.dtype))
            let scaled =
                (expert.asType(.float32) * postScale.weight.asType(expert.dtype).asType(.float32))
                .asType(expert.dtype)
            let normalized = MLXFast.rmsNorm(scaled, weight: .mlxNone, eps: epsilon).asType(
                .float32)
            let next =
                residual * feedforwardAlpha.asType(.float32)
                + normalized * feedforwardBeta.asType(.float32)
            return (next, expert.dtype)
        }
    }

    final class TextModel: Module {
        @ModuleInfo(key: "embed_tokens") var embedding: Embedding
        let layers: [Block]
        let norm: RMSNorm
        let config: MuseSparkConfiguration.Text
        let fullIndex: Int?
        let slidingIndex: Int?

        init(_ config: MuseSparkConfiguration.Text) {
            self.config = config
            _embedding.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            layers = (0 ..< config.layers).map { Block(config, index: $0) }
            norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsEpsilon)
            fullIndex = config.layerTypes.firstIndex(of: "full_attention")
            slidingIndex = config.layerTypes.firstIndex(of: "sliding_attention")
        }

        func embed(_ tokens: MLXArray) -> MLXArray {
            MLXFast.rmsNorm(embedding(tokens), weight: .mlxNone, eps: config.rmsEpsilon)
        }

        func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
            var stream = embeddings.asType(.float32)
            var dtype = DType.bfloat16
            let fullMask = createAttentionMask(h: stream, cache: fullIndex.flatMap { cache?[$0] })
            let windowMask =
                slidingIndex.map {
                    createAttentionMask(
                        h: stream, cache: cache?[$0], windowSize: config.slidingWindow)
                } ?? .none
            for (index, layer) in layers.enumerated() {
                (stream, dtype) = layer(
                    stream, dtype: dtype, mask: config.isSliding(index) ? windowMask : fullMask,
                    cache: cache?[index])
            }
            return norm(stream).asType(.bfloat16)
        }
    }

    final class Model: Module {
        let model: TextModel
        @ModuleInfo(key: "lm_head") var head: Linear
        let config: MuseSparkConfiguration.Text

        init(_ config: MuseSparkConfiguration.Text) {
            self.config = config
            model = TextModel(config)
            _head.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }

        func callAsFunction(
            _ embeddings: MLXArray, cache: [KVCache]?, lastOnly: Bool = false
        ) -> MLXArray {
            var out = model(embeddings, cache: cache)
            if lastOnly { out = out[0..., (-1)..., 0...] }
            if head.weight.dtype.isFloatingPoint { out = out.asType(head.weight.dtype) }
            let logits = head(out).asType(.float32) * config.outputMultiplier
            return config.logitCap > 0 ? tanh(logits / config.logitCap) * config.logitCap : logits
        }
    }
}
