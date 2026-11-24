// Copyright © 2026 Apple Inc.

/// A protocol-neutral semantic event decoded from a model's generated token stream.
public enum ModelOutputEvent: Sendable, Equatable {
    case reasoning(String)
    case response(String)
    case toolCall(ToolCall)
    /// A framed protocol rejected malformed output. Nothing is logged automatically.
    case protocolError(String)
    /// A tool-call-shaped model output rejected by parsing or authorization.
    case rejectedToolCall(RejectedToolCall)
    case stop
}

/// Decodes model-specific token streams into response and tool-call events.
///
/// The generation loop owns iteration and cancellation. A decoder owns the
/// response protocol: token framing, semantic stop tokens, and event routing.
/// This keeps model-specific protocols out of the generic evaluation loop.
protocol TokenStreamDecoder {
    /// Semantic boundaries in addition to the model's ordinary EOS tokens.
    var additionalStopTokenIDs: Set<Int> { get }

    /// Whether semantic stop tokens must be passed through `push` before the
    /// generation loop terminates.
    var receivesStopTokens: Bool { get }

    /// Whether the decoder is currently consuming private reasoning payload.
    /// Sample this before feeding a token to attribute usage accurately.
    var isInsideReasoning: Bool { get }

    /// Tool-call-shaped outputs rejected so far during this generation.
    ///
    /// Decoders whose response protocol never rejects tool calls report zero.
    var rejectedToolCallCount: Int { get }

    /// Calls promoted by bounded cross-dialect recovery during this generation.
    var recoveredToolCallCount: Int { get }

    /// Consumes one generated token. Returns `false` when decoding should stop
    /// because of either a semantic boundary or consumer termination.
    mutating func push(_ token: Int, emit: (ModelOutputEvent) -> Bool) -> Bool

    /// Flushes any buffered events at the end of generation. Returns `false`
    /// when decoding stopped before all buffered events were delivered.
    mutating func finish(emit: (ModelOutputEvent) -> Bool) -> Bool
}

extension TokenStreamDecoder {
    var additionalStopTokenIDs: Set<Int> { [] }
    var receivesStopTokens: Bool { false }
    var isInsideReasoning: Bool { false }
    var rejectedToolCallCount: Int { 0 }
    var recoveredToolCallCount: Int { 0 }
}

/// Decoder for ordinary detokenized tool-call syntaxes.
struct StandardTokenStreamDecoder: TokenStreamDecoder {
    private var detokenizer: StreamingDetokenizer
    private let toolCallProcessor: ToolCallProcessor
    private var stopStringFilter: StopStringFilter
    private var reasoning: ReasoningSegmenter?
    private let tokenizer: any Tokenizer
    private let reasoningControls: (start: Int, end: Int)?
    private var inReasoning = false
    var isInsideReasoning: Bool { reasoning?.inside ?? false }

    init(
        tokenizer: any Tokenizer,
        format: ToolCallFormat,
        tools: [[String: any Sendable]]?,
        stopStrings: Set<String>,
        toolCallPolicy: ToolCallPolicy = .init(),
        reasoning: ReasoningConfig? = nil, reasoningPrimed: Bool = false
    ) {
        self.tokenizer = tokenizer
        if var config = reasoning,
            let start = tokenizer.convertTokenToId(config.startDelimiter),
            let end = tokenizer.convertTokenToId(config.endDelimiter), start != end
        {
            reasoningControls = (start, end)
            // Canonical boundaries are token IDs; literal marker-shaped argument
            // text must not become reasoning. Implicit text exits remain supported.
            config.startDelimiter = ""
            config.endDelimiter = ""
            self.reasoning = ReasoningSegmenter(config: config, primed: reasoningPrimed)
        } else {
            reasoningControls = nil
            self.reasoning = reasoning.map {
                ReasoningSegmenter(config: $0, primed: reasoningPrimed)
            }
        }
        self.detokenizer = StreamingDetokenizer(tokenizer: tokenizer)
        self.toolCallProcessor = ToolCallProcessor(
            format: format, tools: tools, toolCallPolicy: toolCallPolicy)
        self.stopStringFilter = StopStringFilter(stopStrings: stopStrings)
    }

    var rejectedToolCallCount: Int { toolCallProcessor.rejectedToolCallCount }
    var recoveredToolCallCount: Int { toolCallProcessor.recoveredToolCallCount }

    mutating func push(_ token: Int, emit: (ModelOutputEvent) -> Bool) -> Bool {
        if let controls = reasoningControls, token == controls.start || token == controls.end {
            let pending = reasoning?.transition(inside: token == controls.start) ?? []
            detokenizer = StreamingDetokenizer(tokenizer: tokenizer)
            return route(pending, emit: emit)
        }
        detokenizer.append(token: token)
        guard let chunk = detokenizer.next() else { return true }

        return route(reasoning?.push(chunk) ?? [.response(chunk)], emit: emit)
    }

    private mutating func route(_ events: [ModelOutputEvent], emit: (ModelOutputEvent) -> Bool)
        -> Bool
    {
        for event in events {
            switch event {
            case .reasoning:
                if !inReasoning {
                    if let tail = stopStringFilter.finish(),
                        !emitOutputs(toolCallProcessor.processChunkOutputs(tail), emit: emit)
                    {
                        return false
                    }
                    guard emitOutputs(toolCallProcessor.processEOSOutputs(), emit: emit) else {
                        return false
                    }
                }
                inReasoning = true
                guard emit(event) else { return false }
            case .response(let text):
                inReasoning = false
                guard response(text, emit: emit) else { return false }
            default: break
            }
        }
        return true
    }

    private mutating func response(_ chunk: String, emit: (ModelOutputEvent) -> Bool) -> Bool {
        let result = stopStringFilter.process(chunk)
        if let text = result.text,
            !emitOutputs(toolCallProcessor.processChunkOutputs(text), emit: emit)
        {
            return false
        }
        if result.stopped {
            _ = emit(.stop)
            return false
        }
        return true
    }

    mutating func finish(emit: (ModelOutputEvent) -> Bool) -> Bool {
        guard route(reasoning?.finish() ?? [], emit: emit) else { return false }
        if let text = stopStringFilter.finish(),
            !emitOutputs(toolCallProcessor.processChunkOutputs(text), emit: emit)
        {
            return false
        }

        return emitOutputs(toolCallProcessor.processEOSOutputs(), emit: emit)
    }

    /// Maps ordered processor outputs onto stream events, keeping response text,
    /// accepted calls, and rejected calls in the order the model emitted them.
    private func emitOutputs(
        _ outputs: [ToolCallProcessor.Output],
        emit: (ModelOutputEvent) -> Bool
    ) -> Bool {
        for output in outputs {
            let event: ModelOutputEvent =
                switch output {
                case .response(let text): .response(text)
                case .toolCall(let call): .toolCall(call)
                case .rejectedToolCall(let rejection): .rejectedToolCall(rejection)
                }
            guard emit(event) else { return false }
        }
        return true
    }
}
