// Adapted from LiquidAI System One. LFM Open License v1.0;
// see Documentation/Models/LFM3/D1-LICENSE.
import Foundation

/// Options specific to LiquidAI d1's set-attention and score readout.
public struct LFM3DecisionOptions: DecisionModelOptions {
    public enum ScoreReadout: String, Sendable { case median, mean, argmax }
    /// Nil uses the checkpoint's trained group size (16 for the published d1 models).
    public var setSize: Int?
    public var scoreReadout: ScoreReadout
    public init(setSize: Int? = nil, scoreReadout: ScoreReadout = .median) {
        self.setSize = setSize
        self.scoreReadout = scoreReadout
    }
}

struct LFM3DecisionCalibration: Decodable {
    var `default`: Double = 1
    var temperatures: [String: Double] = [:]
    var tables: [String: [[Double]]] = [:]
    enum CodingKeys: String, CodingKey { case `default`, temperatures, tables }
    init() {}
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        `default` = try c.decodeIfPresent(Double.self, forKey: .default) ?? 1
        temperatures = try c.decodeIfPresent([String: Double].self, forKey: .temperatures) ?? [:]
        tables = try c.decodeIfPresent([String: [[Double]]].self, forKey: .tables) ?? [:]
        guard ([`default`] + Array(temperatures.values)).allSatisfy({ $0.isFinite && $0 > 0 })
        else {
            throw ModelFactoryError.invalidConfiguration(
                "d1 calibration temperatures must be positive")
        }
        for knots in tables.values {
            guard
                knots.allSatisfy({
                    $0.count == 2 && $0.allSatisfy { $0.isFinite && (0 ... 1).contains($0) }
                }),
                zip(knots, knots.dropFirst()).allSatisfy({ $0[0] < $1[0] && $0[1] <= $1[1] })
            else { throw ModelFactoryError.invalidConfiguration("Invalid d1 calibration table") }
        }
    }
    func temperature(_ q: DecisionQuestion) -> Double {
        let key: String
        switch q.type {
        case .bool, .noul: key = "noul"
        case .score: key = "score:\(q.criteria.count)"
        default:
            let n = q.criteria.count
            key = "choice:" + (n <= 3 ? "small" : n <= 8 ? "mid" : n <= 26 ? "large" : "inventory")
        }
        return temperatures[key] ?? `default`
    }
    func through(_ name: String, _ value: Double) -> Double {
        guard let knots = tables[name], let first = knots.first, let last = knots.last else {
            return value
        }
        if value <= first[0] { return first[1] }
        if value >= last[0] { return last[1] }
        let i = knots.firstIndex { $0[0] >= value }!
        let (a, b) = (knots[i - 1], knots[i])
        return a[1] + (value - a[0]) * (b[1] - a[1]) / (b[0] - a[0])
    }
    static func softmax(_ values: [Double]) -> [Double] {
        let top = values.max()!
        let exps = values.map { exp($0 - top) }
        let total = exps.reduce(0, +)
        return exps.map { $0 / total }
    }
    func answer(
        _ q: DecisionQuestion, probabilities: [[Double]],
        scoreReadout: LFM3DecisionOptions.ScoreReadout
    ) -> DecisionAnswer {
        if q.type == .multiLabel {
            let p = probabilities.map { through("multiple", $0[0]) }
            return .init(
                type: .multiLabel,
                value: .multiLabel(
                    zip(q.criteria, p).filter { $0.1 >= q.threshold }.map { $0.0.label }),
                probabilities: Dictionary(uniqueKeysWithValues: zip(q.criteria.map(\.label), p)))
        }
        let p = probabilities[0]
        if q.type == .bool || q.type == .noul {
            return .init(type: q.type, value: .bool(p[0] >= 0.5), probability: p[0])
        }
        let best = p.indices.dropFirst().reduce(0) { p[$1] > p[$0] ? $1 : $0 }
        if q.type == .choice {
            return .init(
                type: .choice, value: .choice(q.criteria[best].label),
                probabilities: Dictionary(uniqueKeysWithValues: zip(q.criteria.map(\.label), p)),
                metadata: ["confidence": .double(p[best])])
        }
        var total = 0.0
        let median =
            p.indices.first {
                total += p[$0]
                return total >= 0.5
            } ?? p.count - 1
        let level: Double
        switch scoreReadout {
        case .median: level = Double(median)
        case .argmax: level = Double(best)
        case .mean: level = p.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element }
        }
        return .init(
            type: .score, value: .score(level),
            probabilities: Dictionary(uniqueKeysWithValues: zip(p.indices.map(String.init), p)),
            metadata: [
                "confidence": .double(p[best]),
                "legend": .object(
                    Dictionary(
                        uniqueKeysWithValues:
                            q.criteria.enumerated().map {
                                (
                                    String($0.offset),
                                    .string($0.element.description ?? $0.element.label)
                                )
                            })),
            ])
    }
}

struct LFM3DecisionLayout {
    let text: String
    let offsets: [Int]
    let readouts: [[[Int]]]
}
struct LFM3DecisionRow {
    let tokens: [Int]
    let stateLength: Int
    let groups: [Int]
    let positions: [[Int]]
}
struct LFM3DecisionPrompt {
    let tokenizer: any Tokenizer
    let codes: Int
    let specials: [String]
    func encode(_ text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }
    func plain(_ text: String) -> String {
        var value = text
        // Longest first, repeated for overlapping spellings as in the reference.
        let names = specials.filter { $0.count > 1 && !$0.contains("\u{200b}") }.sorted {
            $0.count > $1.count
        }
        var changed = true
        while changed {
            changed = false
            for name in names where value.contains(name) {
                value = value.replacingOccurrences(
                    of: name, with: String(name.prefix(1)) + "\u{200b}" + name.dropFirst())
                changed = true
            }
        }
        return value
    }
    static func json(_ value: JSONValue, level: Int = 0) throws -> String {
        let indent = String(repeating: "  ", count: level)
        switch value {
        case .object(let object) where !object.isEmpty:
            return "{\n"
                + (try object.keys.sorted().map {
                    try indent + "  " + DecisionJSON.render(.string($0)) + ": "
                        + json(object[$0]!, level: level + 1)
                }).joined(separator: ",\n") + "\n" + indent + "}"
        case .array(let items) where !items.isEmpty:
            return "[\n"
                + (try items.map { try indent + "  " + json($0, level: level + 1) }).joined(
                    separator: ",\n") + "\n" + indent + "]"
        default: return try DecisionJSON.render(value)
        }
    }
    func prefix(state: DecisionState, media: String) throws -> String {
        let body: String
        switch state {
        case .text(let text): body = plain(text + "\n\n") + "\nQUESTION:\n"
        case .json(.null): body = ""
        case .json(let value): body = try plain(Self.json(value) + "\n\n") + "\nQUESTION:\n"
        }
        return (tokenizer.bosToken ?? "") + "<|im_start|>user\n" + media + body
    }
    func aliases(_ labels: [String]) throws -> [(String, Int)] {
        let letters = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZ").map(String.init)
        let native = labels.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        let preferred: [String]
        if native.allSatisfy({ $0.count == 1 && $0.first!.isLetter }) {
            preferred = native
        } else if labels.count <= 26 {
            preferred = Array(letters.prefix(labels.count))
        } else if codes == 2 {
            preferred = Array(
                (letters + letters.flatMap { a in letters.map { a + $0 } }).prefix(labels.count))
        } else {
            preferred = labels.indices.map { String(format: "%02d", $0) }
        }
        let pool =
            letters + (0 ..< 100).map { String(format: "%02d", $0) }
            + letters.map { $0.lowercased() } + (0 ..< 200).map { "#\($0)" }
            + letters.flatMap { a in letters.map { a + $0 } }
        var used = Set<Int>()
        return try preferred.map { code in
            for candidate in [code] + pool {
                let ids = encode(candidate)
                if ids.count == 1, used.insert(ids[0]).inserted { return (candidate, ids[0]) }
            }
            throw ModelExecutionError.invalidOptions(
                "d1 tokenizer has no distinct single-token option code")
        }
    }
    func yesNo() throws -> [[Int]] {
        let result = [["yes", "Yes", "YES"], ["no", "No", "NO"]].map { forms in
            Array(Set(forms.map(encode).filter { $0.count == 1 }.map { $0[0] })).sorted()
        }
        guard result.allSatisfy({ !$0.isEmpty }) else {
            throw ModelExecutionError.invalidOptions("d1 tokenizer needs yes/no tokens")
        }
        return result
    }
    func layout(_ q: DecisionQuestion) throws -> LFM3DecisionLayout {
        var body = q.instructions
        let readouts: [[[Int]]]
        var markers: [String] = []
        switch q.type {
        case .bool, .noul:
            if q.criteria.contains(where: { $0.description != nil }) {
                func description(_ label: String) -> String {
                    q.criteria.first { $0.label == label }?.description ?? "None"
                }
                body += "\nYes: " + description("true") + "\nNo: " + description("false")
            }
            body += "\n\nReply with yes or no only."
            readouts = [try yesNo()]
        case .score:
            guard (2 ... 10).contains(q.criteria.count) else {
                throw ModelExecutionError.invalidOptions("d1 scores need 2 to 10 levels")
            }
            body +=
                "\n\n"
                + q.criteria.enumerated().map {
                    "\($0.offset) \($0.element.description ?? $0.element.label)"
                }.joined(separator: "\n")
                + "\n\nReply with a single digit 0-\(q.criteria.count - 1) only."
            let groups = q.criteria.indices.map { encode(String($0)) }
            guard groups.allSatisfy({ $0.count == 1 }) else {
                throw ModelExecutionError.invalidOptions("d1 scores need single-token digits")
            }
            readouts = [groups]
        case .choice:
            guard q.criteria.count <= 128 else {
                throw ModelExecutionError.invalidOptions("d1 supports at most 128 choice options")
            }
            let names = try aliases(q.criteria.map(\.label))
            body +=
                "\n\nOptions:\n"
                + zip(names, q.criteria).map { code, criterion in
                    let description =
                        criterion.description.flatMap { $0.isEmpty ? nil : $0 }
                        ?? criterion.label.replacingOccurrences(of: "_", with: " ")
                    return code.0 + " " + description
                }.joined(separator: "\n") + "\n\nReply with the option code only."
            readouts = [
                names.map { code, id in
                    let spaced = encode(" " + code)
                    return Array(Set([id] + (spaced.count == 1 ? spaced : []))).sorted()
                }
            ]
        case .multiLabel:
            markers = try q.criteria.map { try DecisionJSON.render(.string($0.label)) }
            body +=
                "\n\nOptions:\n"
                + zip(markers, q.criteria).map { name, criterion in
                    name + ": "
                        + (criterion.description.flatMap { $0.isEmpty ? nil : $0 }
                            ?? criterion.label)
                }.joined(separator: "\n") + "\n\nReply with yes or no for each option."
            readouts = Array(repeating: try yesNo(), count: markers.count)
        }
        var text = plain(body) + "<|im_end|>\n<|im_start|>assistant\n<think></think>"
        if markers.isEmpty { return .init(text: text, offsets: [0], readouts: readouts) }
        var byteEnds: [Int] = []
        for (i, marker) in markers.enumerated() {
            text += (i == 0 ? "" : "\n") + plain(marker) + ":"
            byteEnds.append(text.utf8.count)
        }
        let ids = encode(text)
        let ends = ids.indices.map {
            tokenizer.decode(tokenIds: Array(ids.prefix($0 + 1))).utf8.count
        }
        let offsets = try byteEnds.map { byte -> Int in
            guard let i = ends.firstIndex(where: { $0 >= byte }),
                tokenizer.decode(tokenIds: [ids[i]]).trimmingCharacters(in: .whitespacesAndNewlines)
                    .hasSuffix(":")
            else {
                throw ModelExecutionError.invalidOptions(
                    "d1 answer marker is not an exact colon token")
            }
            return i - ids.count + 1
        }
        return .init(text: text, offsets: offsets, readouts: readouts)
    }
    func pack(prefix: String, layouts: [LFM3DecisionLayout], maxStateTokens: Int) throws
        -> LFM3DecisionRow
    {
        var tokens = encode(prefix + layouts[0].text)
        let state = tokens.count - encode(layouts[0].text).count
        // Truncation can cut media spans and alter the bidirectional state. Fail
        // explicitly instead of silently changing the decision's evidence.
        guard state >= 0, state <= maxStateTokens else {
            throw ModelExecutionError.invalidOptions("d1 state exceeds maxStateTokens")
        }
        var groups =
            Array(repeating: -1, count: state) + Array(repeating: 1, count: tokens.count - state)
        var positions: [[Int]] = []
        for (i, layout) in layouts.enumerated() {
            let length: Int
            if i > 0 {
                let own = encode(
                    (prefix.hasSuffix("QUESTION:\n") ? "QUESTION:\n" : "") + layout.text)
                tokens += own
                groups += Array(repeating: i + 1, count: own.count)
                length = own.count
            } else {
                length = tokens.count - state
            }
            guard layout.offsets.last == 0, Set(layout.offsets).count == layout.offsets.count,
                layout.offsets == layout.offsets.sorted(),
                layout.offsets.allSatisfy({ -length < $0 && $0 <= 0 })
            else { throw ModelExecutionError.invalidOptions("Invalid d1 answer slots") }
            positions.append(layout.offsets.map { tokens.count - 1 + $0 })
        }
        return .init(tokens: tokens, stateLength: state, groups: groups, positions: positions)
    }
}
