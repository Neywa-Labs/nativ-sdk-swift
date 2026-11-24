import Foundation
import MLX
import MLXNN

/// SigLIP's text tower. The embedding reference does not execute its vision tower.
final class SiglipEmbeddingModel: Module, EmbeddingModel {
    struct Configuration: Decodable {
        let vocabularySize: Int
        let hiddenSize: Int
        let intermediateSize: Int
        let heads: Int
        let layers: Int
        let maximumLength: Int
        let epsilon: Float
        let projectionSize: Int
        enum CodingKeys: String, CodingKey {
            case vocabularySize = "vocab_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case heads = "num_attention_heads"
            case layers = "num_hidden_layers"
            case maximumLength = "max_position_embeddings"
            case epsilon = "layer_norm_eps"
            case projectionSize = "projection_size"
        }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 32000
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
            heads = try c.decodeIfPresent(Int.self, forKey: .heads) ?? 12
            layers = try c.decodeIfPresent(Int.self, forKey: .layers) ?? 12
            maximumLength = try c.decodeIfPresent(Int.self, forKey: .maximumLength) ?? 64
            epsilon = try c.decodeIfPresent(Float.self, forKey: .epsilon) ?? 1e-6
            projectionSize = try c.decodeIfPresent(Int.self, forKey: .projectionSize) ?? hiddenSize
            guard vocabularySize > 0, hiddenSize > 0, heads > 0, hiddenSize % heads == 0,
                layers > 0, intermediateSize > 0, maximumLength > 0, projectionSize > 0
            else {
                throw ModelExecutionError.invalidOptions("Invalid SigLIP text configuration")
            }
        }
    }
    final class Embeddings: Module {
        @ModuleInfo(key: "token_embedding") var tokens: Embedding
        @ModuleInfo(key: "position_embedding") var positions: Embedding
        init(_ c: Configuration) {
            _tokens.wrappedValue = Embedding(
                embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
            _positions.wrappedValue = Embedding(
                embeddingCount: c.maximumLength, dimensions: c.hiddenSize)
        }
        func callAsFunction(_ ids: MLXArray) -> MLXArray {
            tokens(ids) + positions(arange(ids.dim(1)).reshaped(1, -1))
        }
    }
    final class Attention: Module {
        @ModuleInfo(key: "q_proj") var q: Linear
        @ModuleInfo(key: "k_proj") var k: Linear
        @ModuleInfo(key: "v_proj") var v: Linear
        @ModuleInfo(key: "out_proj") var output: Linear
        let heads: Int
        let headDim: Int
        init(_ c: Configuration) {
            _q.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
            _k.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
            _v.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
            _output.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
            heads = c.heads
            headDim = c.hiddenSize / c.heads
        }
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            func shape(_ h: MLXArray) -> MLXArray {
                h.reshaped(x.dim(0), x.dim(1), heads, headDim).transposed(0, 2, 1, 3)
            }
            let h = MLXFast.scaledDotProductAttention(
                queries: shape(q(x)), keys: shape(k(x)), values: shape(v(x)),
                scale: pow(Float(headDim), -0.5), mask: .none)
            return output(h.transposed(0, 2, 1, 3).reshaped(x.dim(0), x.dim(1), -1))
        }
    }
    final class MLP: Module {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        init(_ c: Configuration) {
            _fc1.wrappedValue = Linear(c.hiddenSize, c.intermediateSize)
            _fc2.wrappedValue = Linear(c.intermediateSize, c.hiddenSize)
        }
        func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(geluApproximate(fc1(x))) }
    }
    final class Layer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
        @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm
        let mlp: MLP
        init(_ c: Configuration) {
            _attention.wrappedValue = Attention(c)
            _norm1.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.epsilon)
            _norm2.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.epsilon)
            mlp = MLP(c)
        }
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let h = x + attention(norm1(x))
            return h + mlp(norm2(h))
        }
    }
    final class Encoder: Module {
        let layers: [Layer]
        init(_ c: Configuration) { layers = (0 ..< c.layers).map { _ in Layer(c) } }
    }
    final class Transformer: Module {
        let embeddings: Embeddings
        let encoder: Encoder
        @ModuleInfo(key: "final_layer_norm") var norm: LayerNorm
        @ModuleInfo var head: Linear
        init(_ c: Configuration) {
            embeddings = Embeddings(c)
            encoder = Encoder(c)
            _norm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.epsilon)
            _head.wrappedValue = Linear(c.hiddenSize, c.projectionSize)
        }
    }
    final class TextModel: Module {
        @ModuleInfo(key: "text_model") var model: Transformer
        init(_ c: Configuration) { _model.wrappedValue = Transformer(c) }
    }
    @ModuleInfo(key: "text_model") var textModel: TextModel
    let config: Configuration
    var maximumSequenceLength: Int { config.maximumLength }
    init(configuration: Data) throws {
        struct Root: Decodable { let text_config: Configuration }
        config = try JSONDecoder.json5().decode(Root.self, from: configuration).text_config
        _textModel.wrappedValue = TextModel(config)
    }
    func embed(_ input: LMInput) throws -> EmbeddingOutput {
        let (ids, _) = try embeddingTokens(
            input, vocabularySize: config.vocabularySize, maximumLength: maximumSequenceLength)
        let model = textModel.model
        var h = model.embeddings(ids)
        // SigLIP attends to the full fixed-length sequence and pools its final position,
        // including padding. This is a checkpoint convention, unlike masked mean pooling.
        for layer in model.encoder.layers {
            try Task.checkCancellation()
            h = layer(h)
        }
        return .pooled(normalizeEmbeddings(model.head(model.norm(h)[0..., -1, 0...])))
    }
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights
        where !key.hasPrefix("vision_model") && !key.hasPrefix("logit_")
            && !key.contains("position_ids")
        {
            result[
                key.hasPrefix("text_model.") && !key.hasPrefix("text_model.text_model.")
                    ? "text_model." + key : key] = value
        }
        return result
    }
}
