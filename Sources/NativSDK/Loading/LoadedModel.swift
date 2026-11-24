import Foundation
import MLXNN

/// Inference conventions, separate from artifact and checkpoint loading policy.
/// Tool, reasoning, and message contracts are shared with the owned output runtime.
public struct ModelMetadata: Sendable {
    public var extraEOSTokens: Set<String>
    public var stopStrings: Set<String>
    public var eosTokenIds: Set<Int>
    public var toolCallFormat: ToolCallFormat?
    public var reasoningConfig: ReasoningConfig?
    public var messageGenerator: (any MessageGenerator)?

    public init(
        extraEOSTokens: Set<String> = [],
        stopStrings: Set<String>? = nil, eosTokenIds: Set<Int> = [],
        toolCallFormat: ToolCallFormat? = nil, reasoningConfig: ReasoningConfig? = nil,
        messageGenerator: (any MessageGenerator)? = nil
    ) {
        self.extraEOSTokens = extraEOSTokens
        self.stopStrings = stopStrings ?? extraEOSTokens
        self.eosTokenIds = eosTokenIds
        self.toolCallFormat = toolCallFormat
        self.reasoningConfig = reasoningConfig
        self.messageGenerator = messageGenerator
    }
}

/// Exclusively owned loaded resources. This value is neither Copyable nor Sendable.
/// Consume it into a NativModel to serialize complete requests, or into
/// a context for exclusive low-level access. No decoding state is allocated by loading.
struct LoadedModel: ~Copyable {
    let assets: ResolvedModelAssets
    let metadata: ModelMetadata
    let model: any BaseLanguageModel
    let processor: any UserInputProcessor
    let tokenizer: any Tokenizer

    init(
        assets: ResolvedModelAssets, metadata: ModelMetadata,
        model: any BaseLanguageModel, processor: any UserInputProcessor, tokenizer: any Tokenizer
    ) {
        model.train(false)
        self.assets = assets
        self.metadata = metadata
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
    }
}

extension LoadedModel {
    /// Transfers resources to the caller for exclusive low-level access.
    consuming func intoModelContext() -> ModelContext {
        ModelContext(metadata: metadata, model: model, processor: processor, tokenizer: tokenizer)
    }
}

public struct ModelContext {
    public var decisionModel: any DecisionModel {
        get throws {
            guard let model = model as? any DecisionModel else {
                throw ModelExecutionError.unsupportedOperation("predict")
            }
            return model
        }
    }

    /// Requires a generative model; embedding models reject this operation.
    public var languageModel: any LanguageModel {
        get throws {
            guard let model = model as? any LanguageModel else {
                throw ModelExecutionError.unsupportedOperation("generate")
            }
            return model
        }
    }

    public var embeddingModel: any EmbeddingModel {
        get throws {
            guard let model = model as? any EmbeddingModel else {
                throw ModelExecutionError.unsupportedOperation("embed")
            }
            return model
        }
    }

    public var metadata: ModelMetadata
    public var model: any BaseLanguageModel
    public var processor: any UserInputProcessor
    public var tokenizer: Tokenizer

    public init(
        metadata: ModelMetadata, model: any BaseLanguageModel,
        processor: any UserInputProcessor, tokenizer: any Tokenizer
    ) {
        // MLX modules default to training mode. Normalize the complete module tree
        // at the inference boundary. Training APIs enable training for their duration
        // and restore this state afterwards.
        model.train(false)
        self.metadata = metadata
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
    }
}
