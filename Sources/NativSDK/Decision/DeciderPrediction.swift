import Foundation

struct DeciderCalibration: Decodable, Sendable {
    let temperature: Double
    let temperatureByType: [String: Double]
    let isolatedLevels: Bool
    enum CodingKeys: String, CodingKey {
        case temperature
        case temperatureByType = "temperature_by_type"
        case isolatedLevels = "isolated_levels"
    }
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        temperature = try c.decode(Double.self, forKey: .temperature)
        temperatureByType =
            try c.decodeIfPresent([String: Double].self, forKey: .temperatureByType) ?? [:]
        isolatedLevels = try c.decodeIfPresent(Bool.self, forKey: .isolatedLevels) ?? false
        guard
            ([temperature] + Array(temperatureByType.values)).allSatisfy({ $0.isFinite && $0 > 0 })
        else {
            throw ModelExecutionError.invalidOptions(
                "Decision calibration temperatures must be finite and positive")
        }
    }
    func temperature(for type: DecisionType) -> Double {
        temperatureByType[type == .bool || type == .noul ? "noul" : type.rawValue] ?? temperature
    }
}

struct DeciderRow {
    let instruction: String
    let options: [String]
    let temperature: Double
}

struct DeciderPredictionPlan {
    struct Lookup {
        let named: NamedDecisionQuestion
        let start: Int
        let count: Int
        let isolated: Bool
    }
    let rows: [DeciderRow]
    let lookup: [Lookup]

    init(
        questions: [NamedDecisionQuestion], calibration: DeciderCalibration,
        options: DecisionPredictionOptions
    ) throws {
        var rows: [DeciderRow] = []
        var lookup: [Lookup] = []
        let isolate = (options.isolated ?? calibration.isolatedLevels) && options.independent
        for named in questions {
            let q = named.question
            let start = rows.count
            let temperature = calibration.temperature(for: q.type)
            switch q.type {
            case .choice:
                guard (2 ... 255).contains(q.criteria.count) else {
                    throw ModelExecutionError.invalidOptions(
                        "Decider choice needs 2 to 255 options")
                }
                rows.append(
                    .init(
                        instruction: q.instructions,
                        options: q.criteria.map {
                            if let description = $0.description, !description.isEmpty {
                                return $0.label + ": " + description
                            }
                            return $0.label
                        }, temperature: temperature))
            case .score:
                guard (2 ... 10).contains(q.criteria.count) else {
                    throw ModelExecutionError.invalidOptions(
                        "Decider score needs 2 to 10 ordered levels")
                }
                if isolate && q.isolated {
                    for level in q.criteria {
                        let description = (level.description ?? level.label).replacingOccurrences(
                            of: #"^\s*-?\d+\s*:\s*"#, with: "", options: .regularExpression)
                        rows.append(
                            .init(
                                instruction: q.instructions + "\nProposed answer: " + description
                                    + "\nDoes the proposed answer fit?", options: ["no", "yes"],
                                temperature: temperature))
                    }
                } else {
                    rows.append(
                        .init(
                            instruction: q.instructions,
                            options: q.criteria.enumerated().map {
                                "\($0.offset): \($0.element.description ?? $0.element.label)"
                            }, temperature: temperature))
                }
            case .bool, .noul:
                let descriptions = Dictionary(
                    uniqueKeysWithValues: q.criteria.map { ($0.label, $0.description) })
                let labels = [("false", "no"), ("true", "yes")].map { key, label in
                    if let description = descriptions[key] ?? nil, !description.isEmpty {
                        return label + ": " + description
                    }
                    return label
                }
                rows.append(
                    .init(instruction: q.instructions, options: labels, temperature: temperature))
            case .multiLabel:
                throw ModelExecutionError.unsupportedOperation("multi_label decisions")
            }
            lookup.append(
                .init(
                    named: named, start: start, count: rows.count - start,
                    isolated: isolate && q.type == .score && q.isolated))
        }
        self.rows = rows
        self.lookup = lookup
    }

    func answers(probabilities: [[Double]]) throws -> [String: DecisionAnswer] {
        guard probabilities.count == rows.count else {
            throw ModelExecutionError.invalidOptions("Invalid decision probability row count")
        }
        for (row, p) in zip(rows, probabilities) {
            guard p.count == row.options.count, p.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }),
                abs(p.reduce(0, +) - 1) < 1e-4
            else {
                throw ModelExecutionError.invalidOptions(
                    "Invalid decision probability distribution")
            }
        }
        var answers: [String: DecisionAnswer] = [:]
        for item in lookup {
            let q = item.named.question
            var p = probabilities[item.start]
            var metadata: [String: JSONValue] = [:]
            if item.isolated {
                let fit = (0 ..< item.count).map { probabilities[item.start + $0][1] }
                let mass = fit.reduce(0, +)
                p = fit.map { $0 / (mass == 0 ? 1e-9 : mass) }
                metadata["level_fit"] = .object(
                    Dictionary(
                        uniqueKeysWithValues: fit.enumerated().map {
                            (String($0.offset), .double(Self.round($0.element)))
                        }))
                metadata["fit_mass"] = .double(Self.round(mass))
            }
            if q.type == .bool || q.type == .noul {
                let probability = Self.round(p[1])
                answers[item.named.name] = .init(
                    type: .bool, value: .bool(probability >= 0.5), probability: probability)
                continue
            }
            let best = p.indices.max { p[$0] < p[$1] }!
            let entropy = -p.filter { $0 > 0 }.reduce(0) { $0 + $1 * log($1) }
            metadata["confidence"] = .double(Self.round(p[best]))
            metadata["certainty"] = .double(Self.round(max(0, 1 - entropy / log(Double(p.count)))))
            let names = q.type == .choice ? q.criteria.map(\.label) : p.indices.map(String.init)
            let values = Dictionary(uniqueKeysWithValues: zip(names, p.map { Self.round($0) }))
            let value: DecisionValue
            if q.type == .choice {
                value = .choice(names[best])
            } else {
                value = .score(
                    Self.round(
                        p.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element }, digits: 2)
                )
                metadata["legend"] = .object(
                    Dictionary(
                        uniqueKeysWithValues: q.criteria.enumerated().map {
                            (String($0.offset), .string($0.element.description ?? $0.element.label))
                        }))
            }
            answers[item.named.name] = .init(
                type: q.type, value: value, probabilities: values, metadata: metadata)
        }
        return answers
    }
    static func round(_ value: Double, digits: Int = 4) -> Double {
        let scale = pow(10.0, Double(digits))
        return (value * scale).rounded(.toNearestOrEven) / scale
    }
}

struct DeciderPromptBuilder {
    let tokenizer: any Tokenizer
    let labels: [String]
    let labelTokenIDs: [Int]
    let contextTokens: [Int]

    init(tokenizer: any Tokenizer, state: DecisionState, maxStateTokens: Int) throws {
        self.tokenizer = tokenizer
        let alphabet = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZ").map(String.init)
        let candidates = alphabet + alphabet.flatMap { a in alphabet.map { a + $0 } }
        var labels: [String] = []
        var ids: [Int] = []
        for candidate in candidates {
            let tokens = tokenizer.encode(text: candidate, addSpecialTokens: false)
            if tokens.count == 1 {
                labels.append(candidate)
                ids.append(tokens[0])
                if ids.count == 255 { break }
            }
        }
        guard ids.count == 255, Set(ids).count == 255 else {
            throw ModelExecutionError.invalidOptions(
                "The Decider tokenizer must provide 255 distinct label tokens")
        }
        self.labels = labels
        self.labelTokenIDs = ids
        let context: String
        switch state {
        case .text(let text): context = text
        case .json(let value): context = try Self.renderJSON(Self.annotate(value))
        }
        contextTokens = Array(
            tokenizer.encode(text: "Context:\n" + context, addSpecialTokens: false).prefix(
                maxStateTokens))
    }

    func prompt(_ row: DeciderRow) -> [Int] {
        contextTokens + question(row, number: "")
    }
    func packed(_ rows: [DeciderRow]) -> (tokens: [Int], positions: [Int]) {
        var tokens = contextTokens
        var positions: [Int] = []
        for (index, row) in rows.enumerated() {
            tokens += question(row, number: rows.count > 1 ? " \(index + 1)" : "")
            positions.append(tokens.count - 1)
        }
        return (tokens, positions)
    }
    private func question(_ row: DeciderRow, number: String) -> [Int] {
        let head = "\n\nQuestion\(number): \(row.instruction)\nOptions:"
        let tail = "\nAnswer\(number): ("
        if row.options.count <= 10 {
            let body = row.options.enumerated().map { "\n(\(labels[$0.offset])) \($0.element)" }
                .joined()
            return tokenizer.encode(text: head + body + tail, addSpecialTokens: false)
        }
        var tokens = tokenizer.encode(text: head, addSpecialTokens: false)
        let open = tokenizer.encode(text: "\n(", addSpecialTokens: false)
        for (index, value) in row.options.enumerated() {
            tokens +=
                open + [labelTokenIDs[index]]
                + tokenizer.encode(text: ") " + value, addSpecialTokens: false)
        }
        return tokens + tokenizer.encode(text: tail, addSpecialTokens: false)
    }

    static func annotate(_ value: JSONValue) -> JSONValue {
        switch value {
        case .array(let items):
            return .array(
                items.enumerated().map { index, item in
                    let annotated = annotate(item)
                    guard items.count >= 8 else { return annotated }
                    if case .object(let object) = annotated {
                        return .object(["_index": .int(index)].merging(object) { _, new in new })
                    }
                    return .object(["_index": .int(index), "value": annotated])
                })
        case .object(let object): return .object(object.mapValues(annotate))
        default: return value
        }
    }
    /// Stable key order and Python-compatible separators for structured state.
    static func renderJSON(_ value: JSONValue) throws -> String {
        try DecisionJSON.render(value)
    }
}
