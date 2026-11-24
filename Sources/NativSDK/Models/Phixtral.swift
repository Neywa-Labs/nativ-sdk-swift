import Foundation
import MLX
import MLXNN

class PhixtralAttention: Module {
    let heads: Int

    @ModuleInfo(key: "Wqkv") var wqkv: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let rope: RoPE

    init(dimensions: Int, heads: Int, rotaryDimensions: Int) {
        self.heads = heads
        _wqkv.wrappedValue = Linear(dimensions, 3 * dimensions)
        _outProj.wrappedValue = Linear(dimensions, dimensions)
        self.rope = RoPE(dimensions: rotaryDimensions, traditional: false)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let parts = split(wqkv(x), parts: 3, axis: -1)
        let (B, L) = (parts[0].dim(0), parts[0].dim(1))

        var queries = parts[0].reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        var keys = parts[1].reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        let values = parts[2].reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        // The scale comes from the head width after the reshape, and the attention runs in
        // float32 as upstream does.
        let scale = pow(Float(queries.dim(-1)), -0.5)
        let output = attentionWithCacheUpdate(
            queries: queries.asType(.float32), keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .asType(values.dtype)
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return outProj(output)
    }
}

class PhixtralMoE: Module {
    let expertsPerToken: Int

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchMLP
    @ModuleInfo var gate: Linear

    init(_ config: PhixtralConfiguration, dimensions: Int, hiddenDimensions: Int) {
        self.expertsPerToken = config.expertsPerToken
        _switchMLP.wrappedValue = SwitchMLP(
            inputDims: dimensions, hiddenDims: hiddenDimensions,
            numExperts: config.localExperts, bias: true)
        _gate.wrappedValue = Linear(config.modelDimensions, config.localExperts, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = gate(x)

        // Top-k over the raw gate logits, and the softmax is taken over the *selected*
        // logits only. Softmaxing first and then selecting gives different weights, since
        // the discarded experts would still have taken a share of the mass.
        let indices = MLX.stopGradient(
            MLX.argPartition(-gates, kth: expertsPerToken - 1, axis: -1)
        )[.ellipsis, ..<expertsPerToken]
        let scores = softmax(takeAlong(gates, indices, axis: -1), axis: -1, precise: true)

        let y = switchMLP(x, indices)
        return (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

class PhixtralBlock: Module {
    @ModuleInfo(key: "mixer") var mixer: PhixtralAttention
    let ln: LayerNorm
    let moe: PhixtralMoE

    init(_ config: PhixtralConfiguration) {
        let dimensions = config.modelDimensions
        _mixer.wrappedValue = PhixtralAttention(
            dimensions: dimensions, heads: config.heads,
            rotaryDimensions: config.rotaryDimensions)
        self.ln = LayerNorm(dimensions: dimensions)
        self.moe = PhixtralMoE(
            config, dimensions: dimensions, hiddenDimensions: dimensions * 4)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        // Parallel: one norm feeds both branches, and both join the residual.
        let h = ln(x)
        return mixer(h, mask: mask, cache: cache) + moe(h) + x
    }
}

public class PhixtralEmbedding: Module {
    @ModuleInfo(key: "wte") var wte: Embedding

    public init(_ config: PhixtralConfiguration) {
        _wte.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.modelDimensions)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { wte(x) }
}

public class PhixtralOutputHead: Module {
    let ln: LayerNorm
    @ModuleInfo var linear: Linear

    public init(_ config: PhixtralConfiguration) {
        self.ln = LayerNorm(dimensions: config.modelDimensions)
        _linear.wrappedValue = Linear(config.modelDimensions, config.vocabularySize)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { linear(ln(x)) }
}

public class PhixtralTransformer: Module {
    @ModuleInfo(key: "embd") var embd: PhixtralEmbedding

    let h: [PhixtralBlock]

    public init(_ config: PhixtralConfiguration) {
        precondition(config.vocabularySize > 0)
        _embd.wrappedValue = PhixtralEmbedding(config)
        self.h = (0 ..< config.layers).map { _ in PhixtralBlock(config) }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var x = embd(inputs)
        let mask = createAttentionMask(h: x, cache: cache?.first)

        for (i, block) in h.enumerated() {
            x = block(x, mask: mask, cache: cache?[i])
        }

        return x
    }
}

public class PhixtralModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let transformer: PhixtralTransformer

    @ModuleInfo(key: "lm_head") var lmHead: PhixtralOutputHead

    let configuration: PhixtralConfiguration

    public init(_ config: PhixtralConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.layers).map { _ in config.heads }
        self.transformer = PhixtralTransformer(config)
        _lmHead.wrappedValue = PhixtralOutputHead(config)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(transformer(inputs, cache: cache))
    }

    /// Stacks each layer's separately shipped experts into one switched tensor.
    ///
    /// A checkpoint carries `moe.mlp.<expert>.fc1.weight` per expert; the runtime wants
    /// `moe.switch_mlp.fc1.weight` with the experts on a leading axis. Quantized
    /// checkpoints carry `scales` and `biases` alongside, and those stack the same way. A
    /// checkpoint already in the switched form is left alone.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        guard weights["transformer.h.0.moe.mlp.0.fc1.weight"] != nil else { return weights }

        var sanitized = weights
        for layer in 0 ..< configuration.layers {
            let prefix = "transformer.h.\(layer)"
            for projection in ["fc1", "fc2"] {
                for field in ["weight", "scales", "biases", "bias"] {
                    guard sanitized["\(prefix).moe.mlp.0.\(projection).\(field)"] != nil
                    else { continue }

                    var stacked: [MLXArray] = []
                    for expert in 0 ..< configuration.localExperts {
                        let key = "\(prefix).moe.mlp.\(expert).\(projection).\(field)"
                        guard let value = sanitized.removeValue(forKey: key) else { break }
                        stacked.append(value)
                    }
                    guard stacked.count == configuration.localExperts else { continue }
                    sanitized["\(prefix).moe.switch_mlp.\(projection).\(field)"] =
                        MLX.stacked(stacked)
                }
            }
        }
        return sanitized
    }
}

public struct PhixtralConfiguration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var modelDimensions: Int
    var heads: Int
    var layers: Int
    var rotaryDimensions: Int
    var expertsPerToken: Int
    var localExperts: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "num_vocab"
        case modelDimensions = "model_dim"
        case heads = "num_heads"
        case layers = "num_layers"
        case rotaryDimensions = "rotary_dim"
        case expertsPerToken = "num_experts_per_tok"
        case localExperts = "num_local_experts"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // Every field here has a default upstream, and the names are Phi-2's rather than
        // the zoo's: num_vocab and model_dim, not vocab_size and hidden_size.
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "phixtral"
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 51200
        modelDimensions = try c.decodeIfPresent(Int.self, forKey: .modelDimensions) ?? 2560
        heads = try c.decodeIfPresent(Int.self, forKey: .heads) ?? 32
        layers = try c.decodeIfPresent(Int.self, forKey: .layers) ?? 32
        rotaryDimensions = try c.decodeIfPresent(Int.self, forKey: .rotaryDimensions) ?? 32
        expertsPerToken = try c.decodeIfPresent(Int.self, forKey: .expertsPerToken) ?? 2
        localExperts = try c.decodeIfPresent(Int.self, forKey: .localExperts) ?? 4
    }
}

extension PhixtralModel: LoRAModel {
    public var loraLayers: [Module] {
        transformer.h
    }
}
