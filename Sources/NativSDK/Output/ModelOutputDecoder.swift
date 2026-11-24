/// Stateful, single-owner interpretation of generated tokens, without a model.
/// Parsed calls are data; this decoder never executes tools. Tokenizer and tool
public final class ModelOutputDecoder {
    private var decoder: any TokenStreamDecoder
    private var finished = false
    public private(set) var isStopped = false
    public var additionalStopTokenIDs: Set<Int> { decoder.additionalStopTokenIDs }
    public var receivesStopTokens: Bool { decoder.receivesStopTokens }
    public var isInsideReasoning: Bool { decoder.isInsideReasoning }
    public var rejectedToolCallCount: Int { decoder.rejectedToolCallCount }
    public var recoveredToolCallCount: Int { decoder.recoveredToolCallCount }

    public init(
        tokenizer: any Tokenizer, format: ToolCallFormat = .json,
        tools: [[String: any Sendable]]? = nil, stopStrings: Set<String> = [],
        toolCallPolicy: ToolCallPolicy = .init(), reasoning: ReasoningConfig? = nil,
        reasoningPrimed: Bool = false
    ) {
        if let framed = format.makeProtocolTokenStreamDecoder(
            tokenizer: tokenizer, tools: tools, stopStrings: stopStrings,
            toolCallPolicy: toolCallPolicy)
        {
            decoder = framed
        } else {
            decoder = StandardTokenStreamDecoder(
                tokenizer: tokenizer, format: format, tools: tools, stopStrings: stopStrings,
                toolCallPolicy: toolCallPolicy, reasoning: reasoning,
                reasoningPrimed: reasoningPrimed)
        }
    }

    /// Feed ordinary tokens, and terminating tokens when `receivesStopTokens` is true.
    /// After a semantic stop, further pushes are ignored. Call finish once at EOS/limit.
    public func push(_ token: Int) -> [ModelOutputEvent] {
        guard !finished, !isStopped else { return [] }
        var events: [ModelOutputEvent] = []
        _ = decoder.push(token) { event in
            events.append(event)
            return event != .stop
        }
        if additionalStopTokenIDs.contains(token), !events.contains(.stop) { events.append(.stop) }
        isStopped = events.contains(.stop)
        return events
    }

    /// Flush incomplete output using the configured parsing/recovery policy.
    /// Idempotent. Cancellation must call cancel instead, so it cannot commit a call.
    public func finish() -> [ModelOutputEvent] {
        guard !finished else { return [] }
        finished = true
        var events: [ModelOutputEvent] = []
        _ = decoder.finish {
            events.append($0)
            return true
        }
        isStopped = isStopped || events.contains(.stop)
        return events
    }

    /// Discard pending output without EOS repair or tool-call promotion.
    public func cancel() { finished = true }
}
