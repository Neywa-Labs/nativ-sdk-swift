import Foundation
import MLX
import MLXNN

struct LayaConfiguration: Decodable {
    let encoder: ModernBertConfiguration
    let decision: LayaCalibration
    let headLayers: Int
    enum CodingKeys: String, CodingKey {
        case encoder = "encoder_config"
        case decision = "decision_config"
        case headLayers = "head_layers"
    }
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        encoder = try c.decode(ModernBertConfiguration.self, forKey: .encoder)
        decision = try c.decode(LayaCalibration.self, forKey: .decision)
        headLayers = try c.decodeIfPresent(Int.self, forKey: .headLayers) ?? 2
        guard encoder.hiddenSize >= 64, encoder.hiddenSize % 64 == 0, headLayers > 0,
            decision.maxLength <= encoder.maximumLength
        else {
            throw ModelExecutionError.invalidOptions("Invalid Laya backbone or head configuration")
        }
    }
}

final class LayaDecisionLayer: Module {
    let attention: MultiHeadAttention
    let norm1: LayerNorm
    let norm2: LayerNorm
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear
    let activation = ReLU()
    init(width: Int) {
        attention = MultiHeadAttention(dimensions: width, numHeads: width / 64, bias: true)
        norm1 = LayerNorm(dimensions: width)
        norm2 = LayerNorm(dimensions: width)
        _linear1.wrappedValue = Linear(width, 4 * width)
        _linear2.wrappedValue = Linear(4 * width, width)
    }
    func callAsFunction(_ hidden: MLXArray, mask: MLXArray) -> MLXArray {
        let normalized = norm1(hidden)
        let h = hidden + attention(normalized, keys: normalized, values: normalized, mask: mask)
        return h + linear2(activation(linear1(norm2(h))))
    }
}

final class LayaDecisionHead: Module {
    let layers: [LayaDecisionLayer]
    init(width: Int, count: Int) {
        layers = (0 ..< count).map { _ in LayaDecisionLayer(width: width) }
    }
    func callAsFunction(_ hidden: MLXArray, mask: MLXArray) throws -> MLXArray {
        var h = hidden
        for layer in layers {
            try Task.checkCancellation()
            h = layer(h, mask: mask)
        }
        return h
    }
}

final class LayaModel: Module, DecisionModel {
    let encoder: ModernBertEmbeddingModel
    @ModuleInfo(key: "type_emb") var typeEmbedding: Embedding
    let head: LayaDecisionHead
    let scorer: Sequential
    @ModuleInfo(key: "act_head") var actionHead: Sequential
    // Preserve the trained checkpoint tensor; prediction uses decision_config calibration.
    @ParameterInfo var temperature: MLXArray
    let configuration: LayaConfiguration
    let decisionTypes: Set<DecisionType> = [.choice, .score, .bool, .noul]

    init(_ configuration: LayaConfiguration) {
        self.configuration = configuration
        let width = configuration.encoder.hiddenSize
        encoder = ModernBertEmbeddingModel(configuration.encoder, pooling: .cls)
        _typeEmbedding.wrappedValue = Embedding(embeddingCount: 3, dimensions: width)
        head = LayaDecisionHead(width: width, count: configuration.headLayers)
        scorer = Sequential(
            layers: LayerNorm(dimensions: width), Linear(width, width), GELU(), Linear(width, 1))
        _actionHead.wrappedValue = Sequential(
            layers: Linear(width + 4, 256), GELU(), Linear(256, 2))
        _temperature.wrappedValue = MLXArray.ones([3])
    }

    func callAsFunction(
        tokens: MLXArray, attention: MLXArray, markerPositions: MLXArray,
        markerMask: MLXArray, questionTypes: MLXArray
    ) throws -> (logits: MLXArray, action: MLXArray) {
        var hidden = try encoder.encode(tokens, valid: attention .== 1)
        hidden = hidden + typeEmbedding(questionTypes).expandedDimensions(axis: 1)
        let mask = which(
            attention.reshaped(tokens.dim(0), 1, 1, tokens.dim(1)) .== 1,
            MLXArray(Float(0)), MLXArray(Float(-1e9))
        ).asType(hidden.dtype)
        hidden = try head(hidden, mask: mask)
        let batch = arange(hidden.dim(0)).expandedDimensions(axis: 1)
        let markers = hidden[batch, markerPositions]
        let logits = which(
            markerMask, scorer(markers).squeezed(axis: -1).asType(.float32), MLXArray(Float(-1e4)))
        let probabilities = softmax(logits, axis: -1)
        let count = maximum(sum(markerMask.asType(.int32), axis: -1), MLXArray(2)).asType(.float32)
        let entropy =
            -sum(probabilities * log(maximum(probabilities, MLXArray(Float(1e-9)))), axis: -1)
            / log(count)
        let top = sorted(probabilities, axis: -1)[0..., (-2)...]
        let features = stacked(
            [top[0..., 1], top[0..., 1] - top[0..., 0], entropy, count / 255], axis: -1)
        let pooled = hidden[0..., 0].asType(.float32)
        return (logits, actionHead(concatenated([pooled, features], axis: -1)))
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights {
            if let range = key.range(of: ".self_attn.in_proj_") {
                let prefix = key[..<range.lowerBound]
                let suffix = key[range.upperBound...]
                for (name, tensor) in zip(
                    ["query_proj", "key_proj", "value_proj"], value.split(parts: 3, axis: 0))
                {
                    result["\(prefix).attention.\(name).\(suffix)"] = tensor
                }
            } else if key.contains(".self_attn.out_proj.") {
                result[
                    key.replacingOccurrences(
                        of: ".self_attn.out_proj.", with: ".attention.out_proj.")] = value
            } else if key.hasPrefix("scorer.") || key.hasPrefix("act_head."),
                !key.hasPrefix("scorer.layers."), !key.hasPrefix("act_head.layers.")
            {
                let parts = key.split(separator: ".", maxSplits: 2)
                if parts.count == 3 {
                    result["\(parts[0]).layers.\(parts[1]).\(parts[2])"] = value
                } else {
                    result[key] = value
                }
            } else {
                result[key] = value
            }
        }
        return result
    }

    func predict(
        state: DecisionState, questions: [NamedDecisionQuestion], tokenizer: any Tokenizer,
        options: DecisionPredictionOptions
    ) throws -> ModelDecisionResult {
        try DecisionValidation.validate(questions, supported: decisionTypes, options: options)
        guard options.independent, options.isolated == nil else {
            throw ModelExecutionError.unsupportedOperation(
                "Laya packed or isolated-level prediction")
        }
        let builder = try LayaPromptBuilder(
            tokenizer: tokenizer, configuration: configuration.decision)
        let rows = try questions.map {
            try builder.row(state: state, named: $0, maxStateTokens: options.maxStateTokens)
        }
        let length = rows.map { $0.tokens.count }.max()!
        let count = rows.map { $0.markers.count }.max()!
        var tokens: [Int32] = []
        var attention: [Int32] = []
        var positions: [Int32] = []
        var valid: [Bool] = []
        for row in rows {
            guard
                (row.tokens + [builder.padID]).allSatisfy({
                    (0 ..< configuration.encoder.vocabularySize).contains($0)
                })
            else {
                throw ModelExecutionError.invalidOptions(
                    "Laya tokenizer produced out-of-vocabulary tokens")
            }
            tokens +=
                row.tokens.map(Int32.init)
                + Array(repeating: Int32(builder.padID), count: length - row.tokens.count)
            attention +=
                Array(repeating: 1, count: row.tokens.count)
                + Array(repeating: 0, count: length - row.tokens.count)
            positions +=
                row.markers.map(Int32.init) + Array(repeating: 0, count: count - row.markers.count)
            valid +=
                Array(repeating: true, count: row.markers.count)
                + Array(repeating: false, count: count - row.markers.count)
        }
        try Task.checkCancellation()
        let output = try self(
            tokens: MLXArray(tokens, [rows.count, length]),
            attention: MLXArray(attention, [rows.count, length]),
            markerPositions: MLXArray(positions, [rows.count, count]),
            markerMask: MLXArray(valid, [rows.count, count]),
            questionTypes: MLXArray(rows.map { Int32($0.typeIndex) }))
        let action = softmax(output.action.asType(.float32), axis: -1)
        eval(output.logits, action)
        try Task.checkCancellation()
        var answers: [String: DecisionAnswer] = [:]
        for (index, row) in rows.enumerated() {
            let temperature = configuration.decision.temperature(
                type: row.typeIndex, count: row.markers.count)
            let probabilities = softmax(
                output.logits[index, 0 ..< row.markers.count] / Float(temperature)
            ).asArray(Float.self).map(Double.init)
            answers[row.named.name] = try row.answer(
                probabilities: probabilities,
                actProbability: Double(action[index, 0].item(Float.self)))
        }
        return .init(answers: answers, inputTokenCount: rows.reduce(0) { $0 + $1.tokens.count })
    }
}
