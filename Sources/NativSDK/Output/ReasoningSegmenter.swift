import Foundation

/// Lossless between delimiters: whitespace and partial delimiters are retained.
/// Text markers are an interpretation convention; framed protocols use token IDs.
struct ReasoningSegmenter {
    private let config: ReasoningConfig
    private(set) var inside: Bool
    private var pending = ""

    init(config: ReasoningConfig, primed: Bool) {
        self.config = config
        self.inside = primed
    }

    mutating func push(_ text: String) -> [ModelOutputEvent] {
        pending += text
        var events: [ModelOutputEvent] = []
        while !pending.isEmpty {
            let delimiters =
                inside
                ? [config.endDelimiter] + config.implicitEndDelimiters : [config.startDelimiter]
            let matches = delimiters.filter { !$0.isEmpty }.compactMap { delimiter in
                pending.range(of: delimiter).map { (delimiter, $0) }
            }
            if let (delimiter, range) = matches.min(by: { $0.1.lowerBound < $1.1.lowerBound }) {
                append(String(pending[..<range.lowerBound]), to: &events)
                let implicit = inside && delimiter != config.endDelimiter
                pending = String(pending[(implicit ? range.lowerBound : range.upperBound)...])
                inside.toggle()
            } else {
                // Compare scalars so a combining mark cannot hide a delimiter prefix.
                let scalars = Array(pending.unicodeScalars)
                var keep = 0
                for delimiter in delimiters where !delimiter.isEmpty {
                    let marker = Array(delimiter.unicodeScalars)
                    let maximum = min(scalars.count, marker.count - 1)
                    if maximum > 0 {
                        for length in stride(from: maximum, through: 1, by: -1) {
                            if scalars.suffix(length).elementsEqual(marker.prefix(length)) {
                                keep = max(keep, length)
                                break
                            }
                        }
                    }
                }
                append(String(String.UnicodeScalarView(scalars.dropLast(keep))), to: &events)
                pending = String(String.UnicodeScalarView(scalars.suffix(keep)))
                break
            }
        }
        return events
    }

    mutating func finish() -> [ModelOutputEvent] {
        var events: [ModelOutputEvent] = []
        append(pending, to: &events)
        pending = ""
        return events
    }

    mutating func transition(inside: Bool) -> [ModelOutputEvent] {
        let events = finish()
        self.inside = inside
        return events
    }

    private func append(_ text: String, to events: inout [ModelOutputEvent]) {
        guard !text.isEmpty else { return }
        events.append(inside ? .reasoning(text) : .response(text))
    }
}
