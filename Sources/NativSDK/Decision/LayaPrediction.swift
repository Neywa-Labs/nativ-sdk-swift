import Foundation

struct LayaCalibration: Decodable {
    let maxLength: Int
    let headMaxLength: Int
    let temperatures: [JSONValue]
    let temperatureByOptions: [String: JSONValue]
    enum CodingKeys: String, CodingKey {
        case maxLength = "max_len"
        case headMaxLength = "head_max_len"
        case temperatures = "temperature"
        case temperatureByOptions = "temperature_by_options"
    }
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        maxLength = try c.decode(Int.self, forKey: .maxLength)
        headMaxLength = try c.decode(Int.self, forKey: .headMaxLength)
        temperatures = try c.decode([JSONValue].self, forKey: .temperatures)
        temperatureByOptions =
            try c.decodeIfPresent([String: JSONValue].self, forKey: .temperatureByOptions) ?? [:]
        guard maxLength > 0, maxLength <= Int(Int32.max), headMaxLength > 0,
            headMaxLength <= maxLength, temperatures.count >= 3
        else {
            throw ModelExecutionError.invalidOptions("Invalid Laya decision configuration")
        }
    }
    func temperature(type: Int, count: Int) -> Double {
        let bucket = count <= 2 ? "2" : count <= 5 ? "3-5" : count <= 10 ? "6-10" : "11+"
        let key = ["choice", "score", "noul"][type] + ":" + bucket
        let value = temperatureByOptions[key] ?? temperatures[type]
        let temperature: Double
        switch value {
        case .double(let number): temperature = number
        case .int(let number): temperature = Double(number)
        case .string(let string): temperature = Double(string) ?? 1
        case .bool(let bool): temperature = bool ? 1 : 0
        default: temperature = 1
        }
        return temperature.isFinite ? min(5, max(0.5, temperature)) : 1
    }
}

struct LayaRow {
    let named: NamedDecisionQuestion
    let typeIndex: Int
    let labels: [String]
    let descriptions: [String]
    let tokens: [Int]
    let markers: [Int]

    func answer(probabilities p: [Double], actProbability: Double) throws -> DecisionAnswer {
        guard p.count == labels.count, p.allSatisfy({ $0.isFinite && (0 ... 1).contains($0) }),
            abs(p.reduce(0, +) - 1) < 1e-4, actProbability.isFinite,
            (0 ... 1).contains(actProbability)
        else {
            throw ModelExecutionError.invalidOptions("Invalid Laya prediction probabilities")
        }
        var metadata: [String: JSONValue] = [
            "rl_agent": .object(["act_probability": .double(actProbability)])
        ]
        if typeIndex == 2 {
            let probability = Self.round(p[1])
            return .init(
                type: .bool, value: .bool(probability >= 0.5), probability: probability,
                metadata: metadata)
        }
        let entropy = p.reduce(0) { $0 + $1 * log(min(1, max(1e-12, $1))) }
        metadata["confidence"] = .double(Self.round(1 + entropy / log(Double(p.count))))
        let best = p.indices.max { p[$0] < p[$1] }!
        let value: DecisionValue
        if typeIndex == 0 {
            value = .choice(labels[best])
        } else {
            value = .score(
                Self.round(p.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element }))
            metadata["legend"] = .object(
                Dictionary(uniqueKeysWithValues: zip(labels, descriptions.map(JSONValue.string))))
        }
        return .init(
            type: named.question.type, value: value,
            probabilities: Dictionary(uniqueKeysWithValues: zip(labels, p.map(Self.round))),
            metadata: metadata)
    }
    private static func round(_ value: Double) -> Double {
        (value * 10000).rounded(.toNearestOrEven) / 10000
    }
}

struct LayaPromptBuilder {
    let tokenizer: any Tokenizer
    let configuration: LayaCalibration
    let mask: String
    let maskID: Int
    let clsID: Int
    let sepID: Int
    let padID: Int
    init(tokenizer: any Tokenizer, configuration: LayaCalibration) throws {
        self.tokenizer = tokenizer
        self.configuration = configuration
        mask = tokenizer.maskToken ?? "[MASK]"
        guard let maskID = tokenizer.convertTokenToId(mask),
            let clsID = tokenizer.convertTokenToId(tokenizer.clsToken ?? "[CLS]"),
            let sepID = tokenizer.convertTokenToId(tokenizer.sepToken ?? "[SEP]"),
            let padID = tokenizer.convertTokenToId(tokenizer.padToken ?? "[PAD]")
        else {
            throw ModelExecutionError.invalidOptions(
                "Laya requires mask, CLS, SEP, and padding tokens")
        }
        self.maskID = maskID
        self.clsID = clsID
        self.sepID = sepID
        self.padID = padID
    }

    func row(state: DecisionState, named: NamedDecisionQuestion, maxStateTokens: Int) throws
        -> LayaRow
    {
        let q = named.question
        let type: Int
        let labels: [String]
        let descriptions: [String]
        let options: [String]
        switch q.type {
        case .choice:
            type = 0
            labels = q.criteria.map(\.label)
            descriptions = q.criteria.map { $0.description ?? $0.label }
            options = q.criteria.map {
                guard let description = $0.description, !description.isEmpty else {
                    return $0.label
                }
                return $0.label + ": " + description
            }
        case .score:
            type = 1
            labels = q.criteria.indices.map(String.init)
            descriptions = q.criteria.map { $0.description ?? $0.label }
            options = descriptions.enumerated().map { "level \($0.offset): \($0.element)" }
        case .bool, .noul:
            type = 2
            labels = ["false", "true"]
            descriptions = ["no, the statement does not hold", "yes, the statement holds"]
            options = labels.enumerated().map { index, label in
                let description = q.criteria.first { $0.label == label }?.description
                return label + ": "
                    + ((description?.isEmpty == false ? description : nil) ?? descriptions[index])
            }
        case .multiLabel: throw ModelExecutionError.unsupportedOperation("multi_label decisions")
        }
        func encode(_ text: String) -> [Int] {
            tokenizer.encode(
                text: text.replacingOccurrences(of: mask, with: " "), addSpecialTokens: false)
        }
        let head = encode("\(["choice", "score", "noul"][type]) question: \(q.instructions)")
        var optionIDs = options.map { [maskID] + Array(encode(" " + $0).prefix(48)) }
        var budget = configuration.headMaxLength - optionIDs.reduce(0) { $0 + $1.count }
        if budget < 16 {
            let per = max(4, (configuration.headMaxLength - 16) / optionIDs.count)
            optionIDs = optionIDs.map { Array($0.prefix(per)) }
            budget = configuration.headMaxLength - optionIDs.reduce(0) { $0 + $1.count }
        }
        var tokens = [clsID] + Array(head.prefix(max(8, budget))) + [sepID]
        var markers: [Int] = []
        for option in optionIDs {
            markers.append(tokens.count)
            tokens += option
        }
        tokens.append(sepID)
        let room = min(maxStateTokens, max(0, configuration.maxLength - tokens.count - 1))
        let text: String
        switch state {
        case .text(let string): text = string
        case .json(let value): text = try DecisionJSON.render(value)
        }
        tokens += encode(text).prefix(room)
        tokens.append(sepID)
        guard markers.allSatisfy({ $0 < configuration.maxLength }) else {
            throw ModelExecutionError.invalidOptions("Options do not fit the Laya question budget")
        }
        return .init(
            named: named, typeIndex: type, labels: labels, descriptions: descriptions,
            tokens: Array(tokens.prefix(configuration.maxLength)), markers: markers)
    }
}
