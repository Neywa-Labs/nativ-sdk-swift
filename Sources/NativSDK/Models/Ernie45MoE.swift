import Foundation
import MLX
import MLXNN

class Ernie45MoEAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPELayer

    init(_ args: Ernie45MoEConfiguration) {
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDimensions = args.resolvedHeadDimensions
        self.scale = pow(Float(headDimensions), -0.5)

        _wq.wrappedValue = Linear(
            args.hiddenSize, heads * headDimensions, bias: args.useBias)
        _wk.wrappedValue = Linear(
            args.hiddenSize, kvHeads * headDimensions, bias: args.useBias)
        _wv.wrappedValue = Linear(
            args.hiddenSize, kvHeads * headDimensions, bias: args.useBias)
        _wo.wrappedValue = Linear(
            heads * headDimensions, args.hiddenSize, bias: args.useBias)

        self.rope = initializeRope(
            dims: headDimensions,
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: args.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x).reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

class Ernie45MoEBlockSparse: Module, UnaryLayer {
    let topK: Int
    let usesSigmoidGate: Bool

    @ModuleInfo var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: SwiGLUMLP?

    init(_ args: Ernie45MoEConfiguration) {
        self.topK = args.moeK
        self.usesSigmoidGate = args.moeGateActivation == "sigmoid"

        let expertWidth =
            args.moeIntermediateSize > 0 ? args.moeIntermediateSize : args.intermediateSize

        _gate.wrappedValue = Linear(args.hiddenSize, args.moeExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: expertWidth,
            numExperts: args.moeExperts, bias: args.useBias)

        if args.moeSharedExperts > 0 {
            _sharedExperts.wrappedValue = SwiGLUMLP(
                inputDimensions: args.hiddenSize,
                hiddenDimensions: expertWidth * args.moeSharedExperts,
                bias: args.useBias)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let logits = gate(x).asType(.float32)
        let gates = usesSigmoidGate ? sigmoid(logits) : softmax(logits, axis: -1)

        let indices = MLX.stopGradient(
            MLX.argPartition(-gates, kth: topK - 1, axis: -1)
        )[.ellipsis, ..<topK]
        var scores = takeAlong(gates, indices, axis: -1)

        // The floor is not decoration: under a sigmoid gate the selected weights can sum to
        // effectively zero, where a softmax's selection cannot, and dividing by that would
        // be a division by zero rather than a renormalization.
        scores =
            scores / MLX.maximum(scores.sum(axis: -1, keepDims: true), MLXArray(1e-12 as Float))

        var y = (switchMLP(x, indices) * scores[.ellipsis, .newAxis]).sum(axis: -2)
            .asType(x.dtype)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

class Ernie45MoEBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Ernie45MoEAttention
    @ModuleInfo(key: "mlp") var mlp: UnaryLayer

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let isMixture: Bool

    init(_ args: Ernie45MoEConfiguration, layerIndex: Int) {
        _attention.wrappedValue = Ernie45MoEAttention(args)
        self.isMixture = args.isMixtureLayer(layerIndex)
        _mlp.wrappedValue =
            isMixture
            ? Ernie45MoEBlockSparse(args)
            : SwiGLUMLP(
                inputDimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize, bias: args.useBias)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class Ernie45MoEModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [Ernie45MoEBlock]
    let norm: RMSNorm

    public init(_ args: Ernie45MoEConfiguration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map {
            Ernie45MoEBlock(args, layerIndex: $0)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class Ernie45MoEModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Ernie45MoEModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: Ernie45MoEConfiguration

    public init(_ args: Ernie45MoEConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Ernie45MoEModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        for layer in 0 ..< configuration.hiddenLayers
        where configuration.isMixtureLayer(layer) {
            let prefix = "model.layers.\(layer)"
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for field in ["weight", "scales", "biases", "bias"] {
                    guard sanitized["\(prefix).mlp.experts.0.\(projection).\(field)"] != nil
                    else { continue }

                    var stacked: [MLXArray] = []
                    for expert in 0 ..< configuration.moeExperts {
                        let key = "\(prefix).mlp.experts.\(expert).\(projection).\(field)"
                        guard let value = sanitized.removeValue(forKey: key) else { break }
                        stacked.append(value)
                    }
                    guard stacked.count == configuration.moeExperts else { continue }
                    sanitized["\(prefix).mlp.switch_mlp.\(projection).\(field)"] =
                        MLX.stacked(stacked)
                }
            }
        }

        if configuration.tieWordEmbeddings {
            sanitized = sanitized.filter { $0.key != "lm_head.weight" }
        }
        return sanitized
    }
}

public struct Ernie45MoEConfiguration: Decodable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var intermediateSize: Int
    var maxPositionEmbeddings: Int
    var attentionHeads: Int
    var kvHeads: Int
    var hiddenLayers: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var ropeTheta: Float
    var useBias: Bool
    var tieWordEmbeddings: Bool
    var moeExperts: Int
    var moeLayerStart: Int
    var moeIntermediateSize: Int
    var moeK: Int
    var moeLayerInterval: Int
    var moeSharedExperts: Int
    var moeLayerEnd: Int?
    var headDimensions: Int?
    var moeGateActivation: String

    var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    /// Whether layer `index` gets a mixture.
    ///
    /// A windowed range *and* a stride: at or past the start, at or before the end, and on
    /// the stride counted one-based. The end defaults to the last layer.
    func isMixtureLayer(_ index: Int) -> Bool {
        let end = moeLayerEnd ?? (hiddenLayers - 1)
        return (index + 1) % moeLayerInterval == 0 && index >= moeLayerStart && index <= end
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case hiddenLayers = "num_hidden_layers"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case useBias = "use_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case moeExperts = "moe_num_experts"
        case moeLayerStart = "moe_layer_start_index"
        case moeIntermediateSize = "moe_intermediate_size"
        case moeK = "moe_k"
        case moeLayerInterval = "moe_layer_interval"
        case moeSharedExperts = "moe_num_shared_experts"
        case moeLayerEnd = "moe_layer_end_index"
        case headDimensions = "head_dim"
        case moeGateActivation = "moe_gate_act"
    }

    /// The window's bounds arrive as a scalar or as a list; a list collapses to its
    /// extreme, which is what upstream takes.
    private static func bound(
        _ container: KeyedDecodingContainer<CodingKeys>, _ key: CodingKeys, takingMinimum: Bool
    ) throws -> Int? {
        if let scalar = try? container.decodeIfPresent(Int.self, forKey: key) {
            return scalar
        }
        guard let list = try container.decodeIfPresent([Int].self, forKey: key),
            !list.isEmpty
        else { return nil }
        return takingMinimum ? list.min() : list.max()
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "ernie4_5_moe"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        useBias = try c.decode(Bool.self, forKey: .useBias)
        tieWordEmbeddings = try c.decode(Bool.self, forKey: .tieWordEmbeddings)
        moeExperts = try c.decode(Int.self, forKey: .moeExperts)
        moeLayerStart = try Self.bound(c, .moeLayerStart, takingMinimum: true) ?? 0
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        moeK = try c.decodeIfPresent(Int.self, forKey: .moeK) ?? 1
        moeLayerInterval = try c.decodeIfPresent(Int.self, forKey: .moeLayerInterval) ?? 1
        moeSharedExperts = try c.decodeIfPresent(Int.self, forKey: .moeSharedExperts) ?? 0
        moeLayerEnd = try Self.bound(c, .moeLayerEnd, takingMinimum: false)
        headDimensions = try c.decodeIfPresent(Int.self, forKey: .headDimensions)
        moeGateActivation =
            try c.decodeIfPresent(String.self, forKey: .moeGateActivation) ?? "softmax"

        // Refused rather than defaulted, as upstream does: a gate activation this does not
        // implement would otherwise route by whichever one happened to be the fallback.
        guard moeGateActivation == "softmax" || moeGateActivation == "sigmoid" else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: c.codingPath,
                    debugDescription:
                        "moe_gate_act '\(moeGateActivation)' is not supported"))
        }
    }
}

extension Ernie45MoEModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
