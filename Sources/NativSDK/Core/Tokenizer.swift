// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import Tokenizers

/// A protocol for tokenizing text into token IDs and decoding token IDs into text.
public protocol Tokenizer: Sendable {
    func encode(text: String, addSpecialTokens: Bool) -> [Int]
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String
    func convertTokenToId(_ token: String) -> Int?
    func convertIdToToken(_ id: Int) -> String?

    var bosToken: String? { get }
    var eosToken: String? { get }
    var unknownToken: String? { get }
    var clsToken: String? { get }
    var sepToken: String? { get }
    var maskToken: String? { get }
    var padToken: String? { get }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int]
}

extension Tokenizer {
    public var clsToken: String? { nil }
    public var sepToken: String? { nil }
    public var maskToken: String? { nil }
    public var padToken: String? { nil }
    public func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    public func decode(tokenIds: [Int]) -> String {
        decode(tokenIds: tokenIds, skipSpecialTokens: false)
    }

    public var eosTokenId: Int? {
        guard let eosToken else { return nil }
        return convertTokenToId(eosToken)
    }

    public var unknownTokenId: Int? {
        guard let unknownToken else { return nil }
        return convertTokenToId(unknownToken)
    }

    public func applyChatTemplate(
        messages: [[String: any Sendable]]
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages, tools: nil, additionalContext: nil)
    }

    public func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages, tools: tools, additionalContext: nil)
    }
}

public enum TokenizerError: LocalizedError {
    case missingChatTemplate

    public var errorDescription: String? {
        switch self {
        case .missingChatTemplate:
            "This tokenizer does not have a chat template."
        }
    }
}

/// A protocol for loading tokenizers from local directories.
public protocol TokenizerLoader: Sendable {
    func load(from directory: URL) async throws -> any Tokenizer
}

public struct DefaultTokenizerLoader: TokenizerLoader {
    public init() {}

    public func load(from directory: URL) async throws -> any Tokenizer {
        let configurations = LanguageModelConfigurationFromHub(modelFolder: directory)
        guard var configuration = try await configurations.tokenizerConfig else {
            throw Tokenizers.TokenizerError.missingConfig
        }
        let data = try await configurations.tokenizerData
        if data.model.type.string() == "Unigram" {
            let declaredClass = configuration.tokenizerClass.string()?
                .replacingOccurrences(of: "Fast", with: "")
            if !["T5Tokenizer", "XLMRobertaTokenizer", "Xlm-RobertaTokenizer"].contains(
                declaredClass ?? "")
            {
                // Generic tokenizer classes otherwise select the BPE implementation.
                var values = configuration.dictionary() ?? [:]
                values["tokenizer_class"] = .init("XLMRobertaTokenizer")
                configuration = Config(values)
            }
        }

        let hfTokenizer: any Tokenizers.Tokenizer
        do {
            hfTokenizer = try Tokenizers.PreTrainedTokenizer(
                tokenizerConfig: configuration, tokenizerData: data)
        } catch Tokenizers.TokenizerError.unsupportedTokenizer {
            hfTokenizer = try Tokenizers.PreTrainedTokenizer(
                tokenizerConfig: configuration, tokenizerData: data, strict: false)
        }
        let specialTokens = ["cls_token", "sep_token", "mask_token", "pad_token"].reduce(
            into: [String: String]()
        ) { values, key in
            let value: Config = configuration[dynamicMember: key]
            let content: Config = value.content
            if let text = value.string() ?? content.string() { values[key] = text }
        }
        return HFTokenizer(hfTokenizer, specialTokens: specialTokens)
    }
}

private struct HFTokenizer: Tokenizer {
    private let hfTokenizer: any Tokenizers.Tokenizer
    private let specialTokens: [String: String]

    init(_ hfTokenizer: any Tokenizers.Tokenizer, specialTokens: [String: String]) {
        self.hfTokenizer = hfTokenizer
        self.specialTokens = specialTokens
    }
    var clsToken: String? { specialTokens["cls_token"] }
    var sepToken: String? { specialTokens["sep_token"] }
    var maskToken: String? { specialTokens["mask_token"] }
    var padToken: String? { specialTokens["pad_token"] }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        hfTokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens: Bool) -> String {
        hfTokenizer.decode(tokens: tokenIDs, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        hfTokenizer.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        hfTokenizer.convertIdToToken(id)
    }

    var bosToken: String? {
        hfTokenizer.bosToken
    }

    var eosToken: String? {
        hfTokenizer.eosToken
    }

    var unknownToken: String? {
        hfTokenizer.unknownToken
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?,
    )
        throws -> [Int]
    {
        do {
            return try hfTokenizer.applyChatTemplate(
                messages: messages,
                tools: tools,
                additionalContext: additionalContext,
            )
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw TokenizerError.missingChatTemplate
        }
    }
}
