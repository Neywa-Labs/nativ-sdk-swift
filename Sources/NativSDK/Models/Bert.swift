import Foundation
import MLX
import MLXNN

struct BertConfiguration: Decodable {
    let robertaPositions: Bool
    let vocabularySize: Int
    let hiddenSize: Int
    let hiddenLayers: Int
    let attentionHeads: Int
    let intermediateSize: Int
    let maxPositionEmbeddings: Int
    var typeVocabularySize = 1
    var layerNormEpsilon: Float = 1e-5
    var paddingTokenID = 1

    enum CodingKeys: String, CodingKey {
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case typeVocabularySize = "type_vocab_size"
        case layerNormEpsilon = "layer_norm_eps"
        case paddingTokenID = "pad_token_id"
        case activation = "hidden_act"
        case modelType = "model_type"
    }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        robertaPositions = try c.decode(String.self, forKey: .modelType) != "bert"
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        typeVocabularySize =
            try c.decodeIfPresent(Int.self, forKey: .typeVocabularySize)
            ?? (robertaPositions ? 1 : 2)
        layerNormEpsilon =
            try c.decodeIfPresent(Float.self, forKey: .layerNormEpsilon)
            ?? (robertaPositions ? 1e-5 : 1e-12)
        paddingTokenID =
            try c.decodeIfPresent(Int.self, forKey: .paddingTokenID) ?? (robertaPositions ? 1 : 0)
        let activation = try c.decodeIfPresent(String.self, forKey: .activation) ?? "gelu"
        guard hiddenSize > 0, attentionHeads > 0, hiddenSize % attentionHeads == 0,
            hiddenLayers > 0, intermediateSize > 0, vocabularySize > 0,
            paddingTokenID >= 0, paddingTokenID < vocabularySize,
            maxPositionEmbeddings > paddingTokenID + 1, typeVocabularySize > 0,
            activation == "gelu"
        else {
            throw ModelExecutionError.invalidOptions(
                "Invalid or unsupported BERT/XLM-RoBERTa configuration")
        }
    }
}

private enum Bert {
    final class Embeddings: Module {
        @ModuleInfo(key: "word_embeddings") var words: Embedding
        @ModuleInfo(key: "position_embeddings") var positions: Embedding
        @ModuleInfo(key: "token_type_embeddings") var types: Embedding
        @ModuleInfo(key: "LayerNorm") var norm: LayerNorm
        let paddingID: Int
        let robertaPositions: Bool
        init(_ c: BertConfiguration) {
            _words.wrappedValue = Embedding(
                embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
            _positions.wrappedValue = Embedding(
                embeddingCount: c.maxPositionEmbeddings, dimensions: c.hiddenSize)
            _types.wrappedValue = Embedding(
                embeddingCount: c.typeVocabularySize, dimensions: c.hiddenSize)
            _norm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEpsilon)
            paddingID = c.paddingTokenID
            robertaPositions = c.robertaPositions
        }
        func callAsFunction(_ ids: MLXArray) -> MLXArray {
            let valid = (ids .!= paddingID).asType(.int32)
            let indices =
                robertaPositions
                ? cumsum(valid, axis: 1) * valid + paddingID : arange(ids.dim(1)).reshaped(1, -1)
            return norm(
                words(ids) + positions(indices) + types(MLXArray.zeros(ids.shape, dtype: .int32)))
        }
    }
    final class SelfAttention: Module {
        @ModuleInfo var query: Linear
        @ModuleInfo var key: Linear
        @ModuleInfo var value: Linear
        let heads: Int
        let dimensions: Int
        init(_ c: BertConfiguration) {
            _query.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
            _key.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
            _value.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
            heads = c.attentionHeads
            dimensions = c.hiddenSize / c.attentionHeads
        }
        func callAsFunction(_ h: MLXArray, mask: MLXArray) -> MLXArray {
            let b = h.dim(0)
            let l = h.dim(1)
            func shape(_ x: MLXArray) -> MLXArray {
                x.reshaped(b, l, heads, dimensions).transposed(0, 2, 1, 3)
            }
            return MLXFast.scaledDotProductAttention(
                queries: shape(query(h)), keys: shape(key(h)),
                values: shape(value(h)), scale: pow(Float(dimensions), -0.5), mask: .array(mask)
            )
            .transposed(0, 2, 1, 3).reshaped(b, l, -1)
        }
    }
    final class Output: Module {
        @ModuleInfo var dense: Linear
        @ModuleInfo(key: "LayerNorm") var norm: LayerNorm
        init(_ c: BertConfiguration, inputSize: Int) {
            _dense.wrappedValue = Linear(inputSize, c.hiddenSize)
            _norm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEpsilon)
        }
        func callAsFunction(_ x: MLXArray, residual: MLXArray) -> MLXArray {
            norm(dense(x) + residual)
        }
    }
    final class Attention: Module {
        @ModuleInfo(key: "self") var attention: SelfAttention
        let output: Output
        init(_ c: BertConfiguration) {
            _attention.wrappedValue = SelfAttention(c)
            output = Output(c, inputSize: c.hiddenSize)
        }
        func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
            output(attention(x, mask: mask), residual: x)
        }
    }
    final class Intermediate: Module {
        @ModuleInfo var dense: Linear
        init(_ c: BertConfiguration) {
            _dense.wrappedValue = Linear(c.hiddenSize, c.intermediateSize)
        }
        func callAsFunction(_ x: MLXArray) -> MLXArray { gelu(dense(x)) }
    }
    final class Layer: Module {
        let attention: Attention
        let intermediate: Intermediate
        let output: Output
        init(_ c: BertConfiguration) {
            attention = Attention(c)
            intermediate = Intermediate(c)
            output = Output(c, inputSize: c.intermediateSize)
        }
        func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
            let h = attention(x, mask: mask)
            return output(intermediate(h), residual: h)
        }
    }
    final class Encoder: Module {
        let layer: [Layer]
        init(_ c: BertConfiguration) { layer = (0 ..< c.hiddenLayers).map { _ in Layer(c) } }
    }
}

final class BertEmbeddingModel: Module, EmbeddingModel {
    private let embeddings: Bert.Embeddings
    private let encoder: Bert.Encoder
    let maximumSequenceLength: Int
    let vocabularySize: Int
    let pooling: EmbeddingPooling
    init(_ config: BertConfiguration, pooling: EmbeddingPooling) {
        embeddings = Bert.Embeddings(config)
        encoder = Bert.Encoder(config)
        maximumSequenceLength =
            config.maxPositionEmbeddings - (config.robertaPositions ? config.paddingTokenID + 1 : 0)
        vocabularySize = config.vocabularySize
        self.pooling = pooling
    }
    func embed(_ input: LMInput) throws -> EmbeddingOutput {
        let (ids, valid) = try embeddingTokens(
            input, vocabularySize: vocabularySize,
            maximumLength: maximumSequenceLength)
        var h = embeddings(ids)
        let mask = embeddingAttentionMask(valid)
        for layer in encoder.layer {
            try Task.checkCancellation()
            h = layer(h, mask: mask)
        }
        return .pooled(normalizeEmbeddings(pooling.apply(h, mask: valid)))
    }
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights {
            let name =
                key.hasPrefix("roberta.")
                ? String(key.dropFirst(8))
                : (key.hasPrefix("bert.") ? String(key.dropFirst(5)) : key)
            guard !name.contains("position_ids"), !name.hasPrefix("pooler."),
                !name.hasPrefix("lm_head."), !name.hasPrefix("cls.")
            else { continue }
            result[name] = value
        }
        return result
    }
}
