// Copyright © 2025 Apple Inc.

import Foundation

public typealias Message = [String: any Sendable]

public enum Chat {
    public struct Message {
        /// The role of the message sender.
        public var role: Role

        /// The content of the message.
        public var content: String

        /// Array of image data associated with the message.
        public var images: [UserInput.Image]

        /// Array of video data associated with the message.
        public var videos: [UserInput.Video]

        /// Array of audio data associated with the message.
        public var audios: [UserInput.Audio]

        /// Tool-call metadata associated with this message.
        public var tool: Tool?

        public struct Tool: Sendable {
            fileprivate enum Storage: Sendable {
                case calls([ToolCall])
                case result(id: String?, name: String?)
            }

            fileprivate let storage: Storage

            fileprivate init(storage: Storage) {
                self.storage = storage
            }

            /// Tool calls emitted by an assistant message.
            public static func calls(_ calls: [ToolCall]) -> Self {
                Self(storage: .calls(calls))
            }

            /// Identifies the assistant tool call answered by a tool message.
            ///
            /// Some chat templates correlate results by `tool_call_id`, while
            /// others (including Onyx) render the function `name` directly.
            public static func result(id: String, name: String? = nil) -> Self {
                Self(storage: .result(id: id, name: name))
            }

            package var calls: [ToolCall]? {
                guard case .calls(let calls) = storage else { return nil }
                return calls
            }
        }

        public init(
            role: Role, content: String,
            images: [UserInput.Image] = [],
            videos: [UserInput.Video] = [],
            audios: [UserInput.Audio] = [],
            tool: Tool? = nil
        ) {
            self.role = role
            self.content = content
            self.images = images
            self.videos = videos
            self.audios = audios
            self.tool = tool
        }

        public static func system(
            _ content: String, images: [UserInput.Image] = [], videos: [UserInput.Video] = []
        ) -> Self {
            Self(role: .system, content: content, images: images, videos: videos)
        }

        public static func assistant(
            _ content: String,
            images: [UserInput.Image] = [],
            videos: [UserInput.Video] = [],
            toolCalls: [ToolCall]? = nil
        ) -> Self {
            Self(
                role: .assistant, content: content, images: images, videos: videos,
                tool: toolCalls.map { .calls($0) })
        }

        public static func user(
            _ content: String,
            images: [UserInput.Image] = [],
            videos: [UserInput.Video] = [],
            audios: [UserInput.Audio] = []
        ) -> Self {
            Self(role: .user, content: content, images: images, videos: videos, audios: audios)
        }

        public static func tool(
            _ content: String, id: String? = nil, name: String? = nil
        ) -> Self {
            let metadata: Tool? =
                if id != nil || name != nil {
                    Tool(storage: .result(id: id, name: name))
                } else {
                    nil
                }
            return Self(role: .tool, content: content, tool: metadata)
        }

        public enum Role: String, Sendable {
            case user
            case assistant
            case system
            case tool
        }
    }
}

/// Protocol for something that can convert structured
/// ``Chat/Message`` into model specific ``Message``
/// (raw dictionary) format.
///
/// Typically this is owned and used by a ``UserInputProcessor``:
///
/// ```swift
/// public func prepare(input: UserInput) async throws -> LMInput {
///     let messages = Qwen2VLMessageGenerator().generate(from: input)
///     ...
/// ```
public protocol MessageGenerator: Sendable {

    /// Generates messages from the input.
    func generate(from input: UserInput) -> [Message]

    /// Returns array of `[String: any Sendable]` aka ``Message``
    func generate(messages: [Chat.Message]) -> [Message]

    /// Returns `[String: any Sendable]`, aka ``Message``.
    func generate(message: Chat.Message) -> Message
}

extension MessageGenerator {

    public func generate(message: Chat.Message) -> Message {
        var dictionary: Message = [
            "role": message.role.rawValue,
            "content": message.content,
        ]

        addToolMetadata(to: &dictionary, for: message)

        return dictionary
    }

    /// Adds tool-call metadata from a structured message to a raw message dictionary.
    public func addToolMetadata(to dictionary: inout Message, for message: Chat.Message) {
        switch message.tool?.storage {
        case .calls(let calls):
            dictionary["tool_calls"] = calls.map { toolCall -> [String: any Sendable] in
                var entry: [String: any Sendable] = [
                    "type": "function",
                    "function": [
                        "name": toolCall.function.name,
                        "arguments": toolCall.function.argumentsObject,
                    ] as [String: any Sendable],
                ]
                if let id = toolCall.id {
                    entry["id"] = id
                }
                return entry
            }
        case .result(let id, let name):
            if let id { dictionary["tool_call_id"] = id }
            if let name { dictionary["name"] = name }
        case nil:
            break
        }
    }

    public func generate(messages: [Chat.Message]) -> [Message] {
        var rawMessages: [Message] = []

        for message in messages {
            let raw = generate(message: message)
            rawMessages.append(raw)
        }

        return rawMessages
    }

    public func generate(from input: UserInput) -> [Message] {
        switch input.prompt {
        case .text(let text):
            generate(messages: [.user(text)])
        case .messages(let messages):
            messages
        case .chat(let messages):
            generate(messages: messages)
        }
    }
}

/// Default implementation of ``MessageGenerator`` that produces `role` and
/// `content`, plus `name`, `tool_call_id`, and `tool_calls` when present.
///
/// ```swift
/// [
///     "role": message.role.rawValue,
///     "content": message.content,
/// ]
/// ```
public struct DefaultMessageGenerator: MessageGenerator {
    public init() {}
}

struct MediaFirstMessageGenerator: MessageGenerator {
    let textOnlyContentAsString: Bool

    init(textOnlyContentAsString: Bool = false) {
        self.textOnlyContentAsString = textOnlyContentAsString
    }

    func generate(from input: UserInput) -> [Message] {
        switch input.prompt {
        case .text(let text):
            [
                generate(
                    message: .user(
                        text, images: input.images, videos: input.videos, audios: input.audios))
            ]
        case .messages(let messages): messages
        case .chat(let messages): generate(messages: messages)
        }
    }

    func generate(message: Chat.Message) -> Message {
        var content = message.images.map { _ in ["type": "image"] }
        content += message.videos.map { _ in ["type": "video"] }
        content += message.audios.map { _ in ["type": "audio"] }
        var dictionary: Message = ["role": message.role.rawValue]
        if content.isEmpty && textOnlyContentAsString {
            dictionary["content"] = message.content
        } else {
            content.append(["type": "text", "text": message.content])
            dictionary["content"] = content
        }
        addToolMetadata(to: &dictionary, for: message)
        return dictionary
    }
}

/// Decodes JSON-string arguments for templates that iterate over argument objects.
struct ObjectToolArgumentsMessageGenerator: MessageGenerator {
    func generate(from input: UserInput) -> [Message] {
        DefaultMessageGenerator().generate(from: input).map { original in
            guard let calls = original["tool_calls"] as? [Message] else { return original }
            var message = original
            message["tool_calls"] = calls.map { original in
                var call = original
                if let function = call["function"] as? Message {
                    call["function"] = withObjectArguments(function)
                    return call
                }
                return withObjectArguments(call)
            }
            return message
        }
    }

    private func withObjectArguments(_ original: Message) -> Message {
        guard let text = original["arguments"] as? String,
            let object = try? JSONDecoder().decode([String: JSONValue].self, from: Data(text.utf8))
        else { return original }
        var result = original
        result["arguments"] = object.mapValues(\.sendableValue)
        return result
    }
}

/// Implementation of ``MessageGenerator`` that produces a
/// `role` and `content` but omits `system` roles.
///
/// ```swift
/// [
///     "role": message.role.rawValue,
///     "content": message.content,
/// ]
/// ```
public struct NoSystemMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(messages: [Chat.Message]) -> [Message] {
        messages
            .filter { $0.role != .system }
            .map { generate(message: $0) }
    }
}
