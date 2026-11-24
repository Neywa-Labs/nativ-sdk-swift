import Foundation

/// Stable keys and Python-style separators for structured decision contexts.
enum DecisionJSON {
    static func render(_ value: JSONValue) throws -> String {
        switch value {
        case .object(let object):
            return "{"
                + (try object.keys.sorted().map { key in
                    try render(.string(key)) + ": " + render(object[key]!)
                }).joined(separator: ", ") + "}"
        case .array(let items):
            return "[" + (try items.map(render)).joined(separator: ", ") + "]"
        default:
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.withoutEscapingSlashes]
            return String(decoding: try encoder.encode(value), as: UTF8.self)
        }
    }
}
