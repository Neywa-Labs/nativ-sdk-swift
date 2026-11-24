// Copyright © 2026 Apple Inc.

import Foundation

public struct RerankerInput: Sendable {
    /// Token IDs passed to the model.
    public var tokenIds: [Int]

    /// Optional segment IDs for BERT-style sentence-pair models.
    public var tokenTypeIds: [Int]?

    /// Optional model-specific marker IDs used to find pooled query and document states.
    public var markerTokenIds: RerankerMarkerTokenIds?

    public init(
        tokenIds: [Int], tokenTypeIds: [Int]? = nil,
        markerTokenIds: RerankerMarkerTokenIds? = nil
    ) {
        self.tokenIds = tokenIds
        self.tokenTypeIds = tokenTypeIds
        self.markerTokenIds = markerTokenIds
    }
}

/// Token IDs that mark query and document representations in listwise reranker prompts.
///
/// Some listwise rerankers, such as Jina reranker v3, score documents by reading hidden
/// states at special query/document marker tokens. The input processor resolves those
/// marker IDs through the tokenizer and stores them here so the model does not depend on
/// hard-coded tokenizer constants.
public struct RerankerMarkerTokenIds: Sendable {
    public var query: Int
    public var document: Int

    public init(query: Int, document: Int) {
        self.query = query
        self.document = document
    }
}

public protocol ListwiseRerankerModel: BaseLanguageModel {
    func score(input: RerankerInput, documentCount: Int) throws -> [Double]
}

public enum RerankerError: LocalizedError, Sendable {
    case emptyPrompt
    case missingSpecialToken(String)
    case unsupportedModel(String)

    public var errorDescription: String? {
        switch self {
        case .emptyPrompt: "Reranker prompt must not be empty."
        case .missingSpecialToken(let token): "Missing required reranker token: \(token)."
        case .unsupportedModel(let message): message
        }
    }
}
