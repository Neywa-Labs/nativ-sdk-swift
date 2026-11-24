// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import NativSDK

@Suite(.serialized)
struct ToolCallPolicyTests {

    @Test("Default processing forwards schema violations but still rejects undeclared tools")
    func defaultProcessing() {
        let tools: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "read",
                    "parameters": [
                        "type": "object", "properties": ["value": ["type": "integer"]],
                        "required": ["value"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable]
            ]
        ]
        for arguments: [String: JSONValue] in [["value": .string("six")], [:]] {
            let call = ToolCall(function: .init(name: "read", arguments: arguments))
            let json = arguments.isEmpty ? "{}" : #"{"value":"six"}"#
            for format: ToolCallFormat in [.json, .lfm2] {
                let processor = ToolCallProcessor(format: format, tools: tools)
                _ = processor.processChunk(
                    "<tool_call>{\"name\":\"read\",\"arguments\":\(json)}</tool_call>")
                processor.processEOS()
                #expect(processor.toolCalls.first?.function == call.function)
                #expect(processor.rejectedToolCalls.isEmpty)
            }
        }
        let processor = ToolCallProcessor(format: .json, tools: tools)
        _ = processor.processChunk(#"<tool_call>{"name":"other","arguments":{}}</tool_call>"#)
        processor.processEOS()
        #expect(processor.toolCalls.isEmpty)
        #expect(processor.rejectedToolCalls.first?.reason == .undeclaredTool)
    }

    @Test(
        "Token replay preserves both policies and rejection telemetry",
        arguments: ToolCallRecoveryPolicy.allCases, ToolCallValidationPolicy.allCases)
    func generation(recovery: ToolCallRecoveryPolicy, validation: ToolCallValidationPolicy)
        async throws
    {
        let policy = ToolCallPolicy(recovery: recovery, validation: validation)
        let parameters: [String: any Sendable] = [
            "type": "object", "properties": ["value": ["type": "integer"]],
            "required": ["value"],
        ]
        let tools: [[String: any Sendable]] = [
            ["function": ["name": "read", "parameters": parameters] as [String: any Sendable]]
        ]
        for native in [true, false] {
            for value in ["6", "six"] {
                do {
                    let text =
                        native
                        ? "<tool_call>{\"name\":\"read\",\"arguments\":{\"value\":\"\(value)\"}}</tool_call>"
                        : "<function=read><parameter=value>\(value)</parameter></function>"
                    let tokenizer = PolicyTokenizer()
                    let tokens = tokenizer.encode(text: text)
                    let decoder = ModelOutputDecoder(
                        tokenizer: tokenizer, format: .json, tools: tools, toolCallPolicy: policy)
                    var events = tokens.flatMap { decoder.push($0) }
                    events += decoder.finish()
                    let parsed = native || recovery != .disabled
                    let accepted = parsed && (value == "6" || validation == .permissive)
                    let calls =
                        events.compactMap {
                            if case .toolCall(let call) = $0 { return call }
                            return nil
                        } as [ToolCall]
                    let rejected =
                        events.compactMap {
                            if case .rejectedToolCall(let call) = $0 { return call }
                            return nil
                        } as [RejectedToolCall]
                    #expect(calls.count == (accepted ? 1 : 0))
                    if accepted {
                        #expect(
                            calls.first?.function.arguments["value"]
                                == (value == "6" ? .int(6) : .string(value)))
                    }
                    #expect(rejected.count == (parsed && !accepted ? 1 : 0))
                    #expect(rejected.allSatisfy { $0.reason == .invalidArguments })
                    #expect(decoder.rejectedToolCallCount == rejected.count)
                    #expect(decoder.recoveredToolCallCount == (!native && accepted ? 1 : 0))
                }
            }
        }
    }
}

/// One Unicode scalar per token makes every syntax boundary a streaming boundary.
private struct PolicyTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.unicodeScalars.map { Int($0.value) }
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(String.UnicodeScalarView(tokenIds.compactMap(Unicode.Scalar.init)))
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { Unicode.Scalar(id).map(String.init) }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
