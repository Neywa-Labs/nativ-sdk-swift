// Copyright © 2025 Apple Inc.

import Foundation

// MARK: - ToolCallParser Protocol

/// Protocol for parsing tool call content from model output.
///
/// Different models use different formats for tool calls. This protocol provides
/// a common interface for parsing tool calls from model output text.
///
/// Reference: https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/tool_parsers
protocol ToolCallParser: Sendable {
    /// All calls in one complete native frame, preserving source order.
    func parseCalls(content: String, tools: [[String: any Sendable]]?) -> [ToolCall]
    /// The start tag that indicates a tool call is beginning.
    /// Returns `nil` for inline formats that don't use wrapper tags.
    var startTag: String? { get }

    /// The end tag that indicates a tool call has ended.
    /// Returns `nil` for inline formats that don't use wrapper tags.
    var endTag: String? { get }

    /// Whether an unframed JSON object is native call syntax for this parser.
    var supportsBareJSON: Bool { get }

    /// Parse the content into a `ToolCall`.
    /// - Parameters:
    ///   - content: The text content to parse (may include tags)
    ///   - tools: Optional tool schemas for type-aware parsing
    /// - Returns: A `ToolCall` if parsing succeeds, `nil` otherwise
    func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall?

    /// Parse remaining buffered content at end-of-sequence.
    ///
    /// Called when generation ends to extract any tool calls still in the buffer.
    /// The default implementation splits on `startTag` (if present) and parses
    /// each segment individually.
    func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall]
}

extension ToolCallParser {
    func parseCalls(content: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        parse(content: content, tools: tools).map { [$0] } ?? []
    }
    var supportsBareJSON: Bool { false }

    func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        if let startTag {
            return
                toolCallBuffer
                .components(separatedBy: startTag)
                .filter { !$0.isEmpty }
                .compactMap { parse(content: $0, tools: tools) }
        } else {
            guard let toolCall = parse(content: toolCallBuffer, tools: tools) else {
                return []
            }
            return [toolCall]
        }
    }
}

extension ToolCallFormat {
    func createParser() -> any ToolCallParser {
        switch self {
        case .json:
            return JSONToolCallParser(
                startTag: "<tool_call>", endTag: "</tool_call>", supportsBareJSON: true)
        case .lfm2:
            return PythonicToolCallParser(
                startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        case .xmlFunction:
            return XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        case .qwen35:
            return Qwen35ToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        case .glm4:
            return GLM4ToolCallParser()
        case .ling3:
            return GLM4ToolCallParser(allowArgumentlessCalls: true)
        case .gemma:
            return GemmaFunctionParser(
                startTag: "<start_function_call>", endTag: "<end_function_call>",
                escapeMarker: "<escape>")
        case .gemma4:
            return GemmaFunctionParser(
                startTag: "<|tool_call>", endTag: "<tool_call|>", escapeMarker: "<|\"|>")
        case .kimiK2:
            return KimiK2ToolCallParser()
        case .minimaxM2:
            return MiniMaxM2ToolCallParser()
        case .atem:
            return ATEMToolCallParser()
        case .inkling2:
            return Inkling2ToolCallParser()
        case .mistral:
            return MistralToolCallParser()
        case .llama3:
            return Llama3ToolCallParser()
        case .gptOSS:
            return JSONToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        }
    }

    /// Builds a decoder only for formats which own a framed token protocol.
    /// Generic callers use this factory and never depend on an Onyx/Harmony
    /// concrete type. Plain text tool syntaxes return `nil` and stay on their
    /// standard detokenized routing path.
    func makeProtocolTokenStreamDecoder(
        tokenizer: any Tokenizer,
        tools: [[String: any Sendable]]?,
        stopStrings: Set<String>,
        toolCallPolicy: ToolCallPolicy = .init()
    ) -> (any TokenStreamDecoder)? {
        switch self {
        case .gptOSS:
            return HarmonyStreamAdapter(
                tokenizer: tokenizer, tools: tools, stopStrings: stopStrings,
                toolCallPolicy: toolCallPolicy)

        case .atem:
            return OnyxStreamAdapter(
                tokenizer: tokenizer, tools: tools, stopStrings: stopStrings,
                toolCallPolicy: toolCallPolicy)

        case .inkling2:
            return Inkling2StreamAdapter(
                tokenizer: tokenizer, tools: tools, stopStrings: stopStrings,
                toolCallPolicy: toolCallPolicy)

        case .json, .lfm2, .xmlFunction, .qwen35, .glm4, .ling3, .gemma, .gemma4, .kimiK2,
            .minimaxM2,
            .mistral, .llama3:
            return nil
        }
    }

    func generateToolCallID() -> String {
        let uuid = UUID().uuidString.replacingOccurrences(of: "-", with: "")

        switch self {
        case .mistral:
            return String(uuid.prefix(9))
        default:
            return "call_" + uuid.lowercased()
        }
    }
}
