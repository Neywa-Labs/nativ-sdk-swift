import Foundation
import MLX
import MLXNN

class OlmoBlock: Module {
    let heads: Int
    let headDimensions: Int
    let scale: Float

    @ModuleInfo(key: "att_proj") var attProj: Linear
    @ModuleInfo(key: "attn_out") var attnOut: Linear
    @ModuleInfo(key: "ff_proj") var ffProj: Linear
    @ModuleInfo(key: "ff_out") var ffOut: Linear

    @ModuleInfo(key: "att_norm") var attNorm: LayerNorm
    @ModuleInfo(key: "ff_norm") var ffNorm: LayerNorm

    let rope: RoPE

    init(_ config: OlmoConfiguration) {
        let dim = config.modelDimensions
        self.heads = config.heads
        self.headDimensions = dim / config.heads
        self.scale = pow(Float(headDimensions), -0.5)

        _attProj.wrappedValue = Linear(dim, 3 * dim, bias: false)
        _attnOut.wrappedValue = Linear(dim, dim, bias: false)
        _ffProj.wrappedValue = Linear(dim, config.resolvedMLPHiddenSize, bias: false)
        // Half the projection's width, because the projection is split in two.
        _ffOut.wrappedValue = Linear(config.resolvedMLPHiddenSize / 2, dim, bias: false)

        // No weight and no bias on any norm here.
        _attNorm.wrappedValue = LayerNorm(dimensions: dim, affine: false)
        _ffNorm.wrappedValue = LayerNorm(dimensions: dim, affine: false)

        self.rope = RoPE(
            dimensions: headDimensions, traditional: config.ropeTraditional,
            base: config.ropeTheta)
    }

    func attend(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let parts = split(attProj(x), parts: 3, axis: -1)

        var queries = parts[0].reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        var keys = parts[1].reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        let values = parts[2].reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return attnOut(output)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attend(attNorm(x), mask: mask, cache: cache)
        // `swiglu(second, first)` upstream: the gate is the second half. Swapping them
        // loads identically and changes every activation.
        let halves = split(ffProj(ffNorm(h)), parts: 2, axis: -1)
        return h + ffOut(silu(halves[1]) * halves[0])
    }
}

public class OlmoTransformer: Module {
    @ModuleInfo(key: "wte") var wte: Embedding

    let blocks: [OlmoBlock]

    @ModuleInfo(key: "ff_out") var ffOut: Linear?
    let norm: LayerNorm

    let weightTying: Bool

    public init(_ config: OlmoConfiguration) {
        precondition(config.embeddingSize > 0)
        self.weightTying = config.weightTying

        // Sized by embedding_size, not vocab_size: the two differ on published configs and
        // the embedding is the larger one.
        _wte.wrappedValue = Embedding(
            embeddingCount: config.embeddingSize, dimensions: config.modelDimensions)
        self.blocks = (0 ..< config.layers).map { _ in OlmoBlock(config) }
        if !config.weightTying {
            _ffOut.wrappedValue = Linear(
                config.modelDimensions, config.embeddingSize, bias: false)
        }
        self.norm = LayerNorm(dimensions: config.modelDimensions, affine: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = wte(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, block) in blocks.enumerated() {
            h = block(h, mask: mask, cache: cache?[i])
        }

        h = norm(h)
        if let ffOut {
            return ffOut(h)
        }
        return wte.asLinear(h)
    }
}

public class OlmoInner: Module {
    public let transformer: OlmoTransformer

    public init(_ config: OlmoConfiguration) {
        self.transformer = OlmoTransformer(config)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        transformer(inputs, cache: cache)
    }
}

public class OlmoModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "model") var model: OlmoInner

    public init(_ config: OlmoConfiguration) {
        // The head projects to embedding_size, so that is the width a sampler sees.
        self.vocabularySize = config.embeddingSize
        self.kvHeads = (0 ..< config.layers).map { _ in config.heads }
        _model.wrappedValue = OlmoInner(config)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        model(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            var name = key
            if name.hasPrefix("language_model.") {
                name = String(name.dropFirst("language_model.".count))
            }
            if !name.hasPrefix("model.") {
                name = "model." + name
            }
            sanitized[name] = value
        }
        return sanitized
    }
}

public struct OlmoConfiguration: Codable, Sendable {
    var modelType: String
    var modelDimensions: Int
    var layers: Int
    var heads: Int
    var vocabularySize: Int
    var embeddingSize: Int
    var mlpHiddenSize: Int?
    var ropeTheta: Float
    var ropeTraditional: Bool
    var mlpRatio: Int
    var weightTying: Bool

    /// `mlp_hidden_size` when given, else `mlp_ratio` times the model width.
    var resolvedMLPHiddenSize: Int {
        mlpHiddenSize ?? (mlpRatio * modelDimensions)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case modelDimensions = "d_model"
        case layers = "n_layers"
        case heads = "n_heads"
        case vocabularySize = "vocab_size"
        case embeddingSize = "embedding_size"
        case mlpHiddenSize = "mlp_hidden_size"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case mlpRatio = "mlp_ratio"
        case weightTying = "weight_tying"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "olmo"
        modelDimensions = try c.decode(Int.self, forKey: .modelDimensions)
        layers = try c.decode(Int.self, forKey: .layers)
        heads = try c.decode(Int.self, forKey: .heads)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        embeddingSize = try c.decode(Int.self, forKey: .embeddingSize)
        mlpHiddenSize = try c.decodeIfPresent(Int.self, forKey: .mlpHiddenSize)
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        mlpRatio = try c.decodeIfPresent(Int.self, forKey: .mlpRatio) ?? 4
        weightTying = try c.decodeIfPresent(Bool.self, forKey: .weightTying) ?? false

        guard modelDimensions % heads == 0 else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: c.codingPath,
                    debugDescription:
                        "d_model \(modelDimensions) must be divisible by n_heads \(heads)"))
        }
    }
}

extension OlmoModel: LoRAModel {
    public var loraLayers: [Module] {
        model.transformer.blocks
    }
}
