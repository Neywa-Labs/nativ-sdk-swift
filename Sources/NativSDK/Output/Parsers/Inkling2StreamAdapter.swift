import Foundation

struct Inkling2ToolCallParser: ToolCallParser {
    let startTag: String? = "<|content_invoke_tool_json|>"
    let endTag: String? = "<|end_message|>"

    private struct Invocation: Decodable {
        let name: String
        let args: [String: JSONValue]
    }

    func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard let invocation = try? JSONDecoder().decode(Invocation.self, from: Data(content.utf8)),
            !invocation.name.isEmpty
        else { return nil }
        return ToolCall(function: .init(name: invocation.name, arguments: invocation.args))
    }
}

/// Inkling 2 emits separate assistant frames for reasoning, answers, and tool invocations.
struct Inkling2StreamAdapter: TokenStreamDecoder {
    private struct Controls {
        let assistant: Int
        let reasoning: Int
        let response: Int
        let invoke: Int
        let end: Int
    }

    private enum Frame {
        case between
        case header
        case reasoning
        case response
        case tool(String)
    }

    private let tokenizer: any Tokenizer
    private let controls: Controls
    private let tools: [[String: any Sendable]]?
    private let allowedTools: Set<String>?
    private let validation: ToolCallValidationPolicy
    private let parser = Inkling2ToolCallParser()
    private var frame = Frame.between
    private var payload: [Int] = []
    private var detokenizer: StreamingDetokenizer
    private var stopFilter: StopStringFilter
    private(set) var rejectedToolCallCount = 0
    let additionalStopTokenIDs: Set<Int>
    let receivesStopTokens = true

    var isInsideReasoning: Bool {
        if case .reasoning = frame { return true }
        return false
    }

    init?(
        tokenizer: any Tokenizer, tools: [[String: any Sendable]]?,
        stopStrings: Set<String>, toolCallPolicy: ToolCallPolicy
    ) {
        guard let assistant = tokenizer.convertTokenToId("<|message_model|>"),
            let reasoning = tokenizer.convertTokenToId("<|content_thinking|>"),
            let response = tokenizer.convertTokenToId("<|content_text|>"),
            let invoke = tokenizer.convertTokenToId("<|content_invoke_tool_json|>"),
            let end = tokenizer.convertTokenToId("<|end_message|>"),
            Set([assistant, reasoning, response, invoke, end]).count == 5
        else { return nil }
        self.tokenizer = tokenizer
        controls = .init(
            assistant: assistant, reasoning: reasoning, response: response, invoke: invoke, end: end
        )
        additionalStopTokenIDs = Set(
            ["<|content_model_end_sampling|>", "<|endoftext|>"].compactMap(
                tokenizer.convertTokenToId))
        self.tools = tools
        allowedTools = tools.map {
            Set(
                $0.compactMap { tool in
                    (tool["function"] as? [String: any Sendable])?["name"] as? String
                        ?? tool["name"] as? String
                })
        }
        validation = toolCallPolicy.validation
        detokenizer = StreamingDetokenizer(tokenizer: tokenizer)
        stopFilter = StopStringFilter(stopStrings: stopStrings)
    }

    mutating func push(_ token: Int, emit: (ModelOutputEvent) -> Bool) -> Bool {
        if additionalStopTokenIDs.contains(token) {
            guard finish(emit: emit) else { return false }
            _ = emit(.stop)
            return false
        }
        if token == controls.assistant {
            guard finish(emit: emit) else { return false }
            frame = .header
            return true
        }
        if token == controls.end {
            defer { reset() }
            if case .tool(let name) = frame { return toolCall(name: name, emit: emit) }
            return true
        }
        switch frame {
        case .header:
            if token == controls.reasoning {
                frame = .reasoning
            } else if token == controls.response {
                frame = .response
            } else if token == controls.invoke {
                frame = .tool(
                    tokenizer.decode(tokenIds: payload).trimmingCharacters(
                        in: .whitespacesAndNewlines))
                payload.removeAll(keepingCapacity: true)
            } else if payload.count < 128 {
                payload.append(token)
            } else {
                reset()
                return emit(.protocolError("Inkling 2 assistant frame header exceeded 128 tokens."))
            }
        case .reasoning, .response:
            detokenizer.append(token: token)
            if let chunk = detokenizer.next(), !chunk.isEmpty {
                if isInsideReasoning { return emit(.reasoning(chunk)) }
                let result = stopFilter.process(chunk)
                if let text = result.text, !emit(.response(text)) { return false }
                if result.stopped {
                    _ = emit(.stop)
                    return false
                }
            }
        case .tool(let name):
            guard payload.count < 65_536 else {
                defer { reset() }
                return reject(.resourceLimitExceeded, name: name, emit: emit)
            }
            payload.append(token)
        case .between:
            break
        }
        return true
    }

    mutating func finish(emit: (ModelOutputEvent) -> Bool) -> Bool {
        defer { reset() }
        if let tail = stopFilter.finish(), !emit(.response(tail)) { return false }
        if case .tool(let name) = frame {
            return reject(.incompleteOutput, name: name, emit: emit)
        }
        return true
    }

    private mutating func reset() {
        frame = .between
        payload.removeAll(keepingCapacity: true)
        detokenizer = StreamingDetokenizer(tokenizer: tokenizer)
    }

    private mutating func reject(
        _ reason: RejectedToolCall.Reason, name: String, detail: String? = nil,
        emit: (ModelOutputEvent) -> Bool
    ) -> Bool {
        rejectedToolCallCount += 1
        return emit(
            .rejectedToolCall(
                .init(
                    reason: reason, format: .inkling2, toolName: name,
                    rawText: tokenizer.decode(tokenIds: payload, skipSpecialTokens: false),
                    detail: detail)))
    }

    private mutating func toolCall(name: String, emit: (ModelOutputEvent) -> Bool) -> Bool {
        let raw = tokenizer.decode(tokenIds: payload, skipSpecialTokens: false)
        guard !name.isEmpty else { return reject(.missingToolName, name: name, emit: emit) }
        guard let call = parser.parse(content: raw, tools: tools), call.function.name == name else {
            return reject(.malformedSyntax, name: name, emit: emit)
        }
        guard allowedTools?.contains(name) ?? true else {
            return reject(.undeclaredTool, name: name, emit: emit)
        }
        let normalized = ToolArgumentNormalization.normalize(call, tools: tools)
        if validation == .strict,
            case .invalid(let violations) = ToolSchemaValidator.validate(
                arguments: normalized.function.arguments, forToolNamed: name, in: tools)
        {
            return reject(
                .invalidArguments, name: name, detail: ToolSchemaValidator.describe(violations),
                emit: emit)
        }
        if let tail = stopFilter.finish(), !emit(.response(tail)) { return false }
        return emit(
            .toolCall(
                .init(
                    function: normalized.function, id: ToolCallFormat.inkling2.generateToolCallID())
            ))
    }
}
