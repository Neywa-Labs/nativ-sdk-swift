import Foundation

public enum DecisionType: String, Codable, Sendable, CaseIterable {
    case choice, score, bool, noul
    case multiLabel = "multi_label"
}

public struct DecisionCriterion: Codable, Sendable, Equatable, ExpressibleByStringLiteral {
    public var label: String
    public var description: String?
    public init(_ label: String, description: String? = nil) {
        self.label = label
        self.description = description
    }
    public init(stringLiteral value: String) { self.init(value) }
}

/// A named question's criteria retain their order, including ordered score levels.
public struct DecisionQuestion: Sendable {
    public var type: DecisionType
    public var instructions: String
    public var criteria: [DecisionCriterion]
    public var isolated: Bool
    public var threshold: Double
    public init(
        type: DecisionType = .choice, instructions: String,
        criteria: [DecisionCriterion] = [], isolated: Bool = true, threshold: Double = 0.5
    ) {
        self.type = type
        self.instructions = instructions
        self.criteria = criteria
        self.isolated = isolated
        self.threshold = threshold
    }
    public static func choice(_ instructions: String, criteria: [DecisionCriterion]) -> Self {
        .init(instructions: instructions, criteria: criteria)
    }
    public static func score(_ instructions: String, levels: [String], isolated: Bool = true)
        -> Self
    {
        .init(
            type: .score, instructions: instructions,
            criteria: levels.enumerated().map { .init(String($0.offset), description: $0.element) },
            isolated: isolated)
    }
    public static func bool(
        _ instructions: String, falseDescription: String? = nil, trueDescription: String? = nil
    ) -> Self {
        .init(
            type: .bool, instructions: instructions,
            criteria: [
                .init("false", description: falseDescription),
                .init("true", description: trueDescription),
            ])
    }
}

/// Use an array of named questions when packed evaluation needs an explicit order.
public struct NamedDecisionQuestion: Sendable {
    public var name: String
    public var question: DecisionQuestion
    public init(_ name: String, question: DecisionQuestion) {
        self.name = name
        self.question = question
    }
}

public enum DecisionState: Sendable, ExpressibleByStringLiteral {
    case text(String)
    case json(JSONValue)
    public init(stringLiteral value: String) { self = .text(value) }
}

/// Additional parameters owned and validated by a decision model family.
public protocol DecisionModelOptions: Sendable {}

/// Ordered media at the head of a decision state; supported kinds depend on the model.
public enum DecisionMedia: Sendable {
    case image(URL)
    case audio(URL)
    case video(URL)
}

public struct DecisionPredictionOptions: Sendable {
    public var modelOptions: (any DecisionModelOptions)?
    public var independent: Bool
    /// nil uses the checkpoint's isolated-level setting; packed evaluation disables isolation.
    public var isolated: Bool?
    /// Truncates only the context prefix; questions and options remain complete.
    public var maxStateTokens: Int
    public init(independent: Bool = true, isolated: Bool? = nil, maxStateTokens: Int = 32768) {
        self.independent = independent
        self.isolated = isolated
        self.maxStateTokens = maxStateTokens
    }
}

public enum DecisionValue: Sendable, Equatable, Codable {
    case choice(String)
    case score(Double)
    case bool(Bool)
    case multiLabel([String])
    public init(from decoder: any Decoder) throws {
        let value = try decoder.singleValueContainer()
        if let bool = try? value.decode(Bool.self) {
            self = .bool(bool)
        } else if let string = try? value.decode(String.self) {
            self = .choice(string)
        } else if let labels = try? value.decode([String].self) {
            self = .multiLabel(labels)
        } else {
            self = .score(try value.decode(Double.self))
        }
    }
    public func encode(to encoder: any Encoder) throws {
        var value = encoder.singleValueContainer()
        switch self {
        case .choice(let string): try value.encode(string)
        case .score(let score): try value.encode(score)
        case .bool(let bool): try value.encode(bool)
        case .multiLabel(let labels): try value.encode(labels)
        }
    }
}

public struct DecisionAnswer: Sendable, Codable, Equatable {
    public let type: DecisionType
    public let value: DecisionValue
    /// Exclusive choice/score probabilities; independent scores belong in metadata.
    public let probabilities: [String: Double]?
    public let probability: Double?
    public let metadata: [String: JSONValue]
    public init(
        type: DecisionType, value: DecisionValue, probabilities: [String: Double]? = nil,
        probability: Double? = nil, metadata: [String: JSONValue] = [:]
    ) {
        self.type = type
        self.value = value
        self.probabilities = probabilities
        self.probability = probability
        self.metadata = metadata
    }
}

public struct ModelDecisionResult: Sendable, Encodable {
    public let answers: [String: DecisionAnswer]
    public let inputTokenCount: Int
    public var outputTokenCount: Int { 0 }
    public init(answers: [String: DecisionAnswer], inputTokenCount: Int) {
        self.answers = answers
        self.inputTokenCount = inputTokenCount
    }
    private enum CodingKeys: String, CodingKey { case answers, usage }
    private struct Usage: Encodable {
        let inputTokens: Int
        let outputTokens: Int
        enum CodingKeys: String, CodingKey {
            case inputTokens = "input_tokens"
            case outputTokens = "output_tokens"
        }
    }
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(answers, forKey: .answers)
        try container.encode(Usage(inputTokens: inputTokenCount, outputTokens: 0), forKey: .usage)
    }
}

/// A model's native prediction contract; implementations retain their own calibration.
public protocol DecisionModel: BaseLanguageModel {
    var decisionTypes: Set<DecisionType> { get }
    func predict(
        state: DecisionState, questions: [NamedDecisionQuestion], tokenizer: any Tokenizer,
        options: DecisionPredictionOptions
    ) throws -> ModelDecisionResult
}

enum DecisionValidation {
    static func validate(
        _ questions: [NamedDecisionQuestion], supported: Set<DecisionType>,
        options: DecisionPredictionOptions
    ) throws {
        guard !questions.isEmpty, options.maxStateTokens > 0,
            Set(questions.map(\.name)).count == questions.count
        else {
            throw ModelExecutionError.invalidOptions(
                "Prediction requires unique named questions and a positive state token limit")
        }
        for named in questions {
            let q = named.question
            guard supported.contains(q.type) else {
                throw ModelExecutionError.unsupportedOperation("\(q.type.rawValue) decisions")
            }
            guard !q.instructions.isEmpty else {
                throw ModelExecutionError.invalidOptions("Question needs instructions")
            }
            if q.type == .choice || q.type == .multiLabel {
                let minimum = q.type == .choice ? 2 : 1
                guard q.criteria.count >= minimum, q.criteria.allSatisfy({ !$0.label.isEmpty }),
                    Set(q.criteria.map(\.label)).count == q.criteria.count
                else {
                    throw ModelExecutionError.invalidOptions(
                        "Decision labels must be nonempty and unique")
                }
            }
            if q.type == .score && q.criteria.count < 2 {
                throw ModelExecutionError.invalidOptions("Score needs at least two ordered levels")
            }
            if q.type == .multiLabel && (!q.threshold.isFinite || !(0 ... 1).contains(q.threshold))
            {
                throw ModelExecutionError.invalidOptions("Threshold must be between zero and one")
            }
            if q.type == .bool || q.type == .noul {
                guard q.criteria.allSatisfy({ $0.label == "false" || $0.label == "true" }),
                    Set(q.criteria.map(\.label)).count == q.criteria.count
                else {
                    throw ModelExecutionError.invalidOptions(
                        "Boolean criteria must describe false and true")
                }
            }
        }
    }
}

/// Preparation can decode media asynchronously; execution stays on the model worker.
protocol PreparedDecisionRequest {
    func execute() throws -> ModelDecisionResult
}
protocol MultimodalDecisionModel: DecisionModel {
    func preparePrediction(
        state: DecisionState, questions: [NamedDecisionQuestion],
        tokenizer: any Tokenizer, options: DecisionPredictionOptions,
        media: [DecisionMedia], isolation: isolated (any Actor)?
    ) async throws -> any PreparedDecisionRequest
}
