// Copyright © 2024, 2025, 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Abstract form of a model that processes language.
public protocol BaseLanguageModel: Module, QuantizationPolicyProviding {
    /// Optionally preprocess the weights and modify / remove values as needed.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]

    /// Optionally preprocess the weights with access to safetensor metadata.
    ///
    /// The default implementation forwards to ``sanitize(weights:)``.
    /// Models can override this to inspect metadata (e.g. check `metadata["format"] == "mlx"`)
    /// and skip or customize sanitization accordingly.
    func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray]
}

/// Weight files a model needs that no naming convention or `model.safetensors.index.json`
/// selects.
///
/// A checkpoint can ship weights in a file that neither the conventional `model*.safetensors`
/// names nor its own index cover: `jinaai/jina-reranker-v3-mlx` keeps its reranking head in
/// `projector.safetensors` and maps only the transformer shards in its index, so the head is
/// never read and the model fails to load. The reference implementation has the same gap and
/// closes it the same way -- the checkpoint's `rerank.py` loads that file by name.
///
/// Conform a model to this protocol to name those files. Being explicit rather than widening
/// the selection is what keeps unrelated weights out: a stray tensor whose name a model's
/// `sanitize(weights:)` rewrites is loaded silently rather than reported.
public protocol AdditionalWeightFilesProviding {
    /// File names, relative to the model directory.
    ///
    /// They are loaded after the selected weight files, so a file that is already selected is
    /// not loaded twice, and names that are not present are ignored.
    var additionalWeightFiles: [String] { get }
}

/// Optional metadata a model wants written into converted safetensors.
///
/// Model-specific metadata lets future loaders distinguish transformed MLX-native
/// checkpoints from original upstream checkpoints without relying only on tensor shapes.
public protocol ModelConversionMetadataProvider {
    var modelConversionMetadata: [String: String] { get }
}

extension BaseLanguageModel {
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        sanitize(weights: weights)
    }
}

/// Time/Height/Width struct to represent information about input images.
public struct THW: Sendable {

    public let t: Int
    public let h: Int
    public let w: Int

    public init(_ t: Int, _ h: Int, _ w: Int) {
        self.t = t
        self.h = h
        self.w = w
    }

    public var values: (Int, Int, Int) {
        (t, h, w)
    }

    public var product: Int { t * h * w }
}

/// Representation of ``LanguageModel`` input.
///
/// This can contain text (tokens), prepared images (`MLXArray`), or other media as
/// needed. ``LMInput`` is produced by ``UserInputProcessor`` in response
/// to ``UserInput``.
///
/// The ``ModelContext`` holds the ``UserInputProcessor`` associated with a
/// ``LanguageModel``.
public struct LMInput {
    public let text: Text
    public let image: ProcessedImage?
    public let video: ProcessedVideo?
    public let audio: ProcessedAudio?

    /// Representation of tokenized input text.
    public struct Text {

        /// input token array
        public let tokens: MLXArray

        /// optional mask array
        public let mask: MLXArray?

        public init(tokens: MLXArray, mask: MLXArray? = nil) {
            self.tokens = tokens
            self.mask = mask
        }

        public subscript(
            indices: MLXArrayIndex..., stream stream: StreamOrDevice = .default
        ) -> Text {
            Text(tokens: tokens[indices, stream: stream], mask: mask?[indices, stream: stream])
        }

        public subscript(
            text indices: MLXArrayIndex..., stream stream: StreamOrDevice = .default
        ) -> Text {
            Text(tokens: tokens[indices, stream: stream], mask: mask)
        }

        /// Per-batch sequence lengths derived from the optional attention mask.
        public var sequenceLengths: [Int]? {
            if let mask {
                return mask.asType(.int32).sum(axis: -1).asArray(Int.self)
            }
            guard tokens.ndim == 2 else { return nil }
            return Array(repeating: tokens.dim(1), count: tokens.dim(0))
        }

        /// Number of logical sequence positions consumed by one model call.
        /// Batch dimensions do not duplicate the shared cache timeline.
        @inline(__always)
        package var cacheSequenceLength: Int {
            tokens.ndim == 0 ? 0 : tokens.dim(-1)
        }
    }

    /// Representation of prepared input image(s).
    public struct ProcessedImage {

        /// Concatenated pixels from one or more images
        public let pixels: MLXArray
        /// Optional per-patch position ids for encoder-free vision embedders.
        public let positionIds: MLXArray?
        /// Time, height, and width of the images
        public let frames: [THW]?
        /// Unpadded image content sizes in pixels, for content-normalized vision encoders.
        public let contentSizes: [CGSize]?

        public init(
            pixels: MLXArray, positionIds: MLXArray? = nil, frames: [THW]? = nil,
            contentSizes: [CGSize]? = nil
        ) {
            self.pixels = pixels
            self.positionIds = positionIds
            self.frames = frames
            self.contentSizes = contentSizes
        }
    }

    /// Representation of prepared input video(s).
    /// For now, this is virtually identical to ProcessedImage.
    public struct ProcessedVideo {

        public let pixels: MLXArray
        public let positionIds: MLXArray?
        public let frames: [THW]?

        public init(
            pixels: MLXArray, positionIds: MLXArray? = nil, frames: [THW]? = nil
        ) {
            self.pixels = pixels
            self.positionIds = positionIds
            self.frames = frames
        }
    }

    /// Representation of prepared audio features.
    public struct ProcessedAudio {
        public let features: MLXArray
        public let mask: MLXArray?

        public init(features: MLXArray, mask: MLXArray? = nil) {
            self.features = features
            self.mask = mask
        }

        public init(samples: MLXArray) {
            self.init(features: samples)
        }
    }

    public init(tokens: MLXArray, mask: MLXArray? = nil) {
        self.init(text: .init(tokens: tokens, mask: mask))
    }

    public init(
        text: LMInput.Text,
        image: LMInput.ProcessedImage? = nil,
        video: LMInput.ProcessedVideo? = nil,
        audio: LMInput.ProcessedAudio? = nil
    ) {
        self.text = text
        self.image = image
        self.video = video
        self.audio = audio
    }
}

/// A model that can split its own prepared input at a token boundary, keeping
/// only the media whose placeholders fall in the suffix.
///
/// Conform only if the media encoder computes each item's features independently
/// of the others; otherwise dropping cached items changes the features of the
/// items that remain. `Qwen25VL` masks each frame to itself and conforms;
/// `Qwen2VL` attends across the whole buffer and does not.
public protocol PreparedInputSplitting {

    /// Return the suffix of `input` that begins at `prefixTokenCount`, carrying only
    /// the media whose placeholder tokens lie inside that suffix.
    ///
    /// Returning `nil` means the input cannot be split safely at this boundary and
    /// the caller must fall back to a full prefill. Implementations should return
    /// `nil` rather than guess -- in particular when the boundary falls inside a
    /// media block, when a payload cannot be attributed to individual media items,
    /// or when the input carries state the implementation does not know how to slice.
    ///
    /// - Parameters:
    ///   - input: the prepared input for the full prompt
    ///   - prefixTokenCount: number of leading tokens already represented in the cache
    /// - Returns: an `LMInput` whose tokens are `input`'s tokens after the first
    ///   `prefixTokenCount`, with a media payload consistent with those tokens, or
    ///   `nil` if no such input can be produced.
    func splitPreparedInput(_ input: LMInput, droppingFirst prefixTokenCount: Int) -> LMInput?
}

/// ``LanguageModel`` step output. This is consumed internally
/// by the ``AutoregressiveIterator``.
public struct LMOutput {

    /// logits (one hot vector of probabilities for tokens)
    public let logits: MLXArray

    /// optional ``State`` to carry forward into the next step
    public let state: State?

    /// typed key for use in ``State``
    public struct Key<T>: Identifiable, Sendable {
        public let id: String

        public init(_ id: String) {
            self.id = id
        }
    }

    /// Dictionary of typed ``Key`` to carry state between steps.
    public struct State {
        private var contents: [String: Any]

        public init() {
            self.contents = [:]
        }

        init(serializedArrays: [String: MLXArray]) {
            self.contents = serializedArrays.mapValues { $0 as Any }
        }

        func serializedArrays() throws -> [String: MLXArray] {
            var arrays: [String: MLXArray] = [:]
            for (key, value) in contents {
                guard let array = value as? MLXArray else {
                    throw SerializationError.unsupportedValue(
                        key: key, type: String(describing: type(of: value)))
                }
                arrays[key] = array
            }
            return arrays
        }

        public subscript<T>(_ key: Key<T>) -> T? {
            get {
                contents[key.id] as? T
            }
            set {
                contents[key.id] = newValue
            }
        }

        enum SerializationError: LocalizedError {
            case unsupportedValue(key: String, type: String)

            var errorDescription: String? {
                switch self {
                case .unsupportedValue(let key, let type):
                    "LMOutput.State key '\(key)' contains unsupported value type '\(type)'"
                }
            }
        }
    }

    public init(logits: MLXArray, state: LMOutput.State? = nil) {
        self.logits = logits
        self.state = state
    }
}

/// Target writes the hidden representation required by its MTP head here.
/// This is architecture-specific, but must match the representation used to
/// train the paired MTP head. Qwen and Gemma assistants both publish their
/// post-final-norm representation. The iterator threads it to the drafter
/// unchanged as `lastHidden`.
public let mtpLastHiddenStatesKey =
    LMOutput.Key<MLXArray>("mtp.lastHiddenStates")

/// Target writes one `(keys, values)` tuple per `layer_type`
/// (`"full_attention"`, `"sliding_attention"`) here, drawn from the last
/// layer of each type. ``MTPSpeculativeTokenIterator`` reads it and threads
/// it to the drafter as `sharedKV`.
public let mtpSharedKVStatesKey =
    LMOutput.Key<[String: (MLXArray, MLXArray)]>("mtp.sharedKVStates")

/// Target writes the absolute cache offset for each emitted shared-K/V
/// layer-type here. The K/V tensors themselves may have a capped sequence
/// axis when backed by a rotating cache, so their shape is not always the
/// absolute RoPE/query position.
public let mtpSharedKVOffsetsKey =
    LMOutput.Key<[String: Int]>("mtp.sharedKVOffsets")

/// Target writes optional position delta state here for MTP drafters that
/// need to reproduce target-specific RoPE continuation positions.
public let mtpPositionDeltasKey =
    LMOutput.Key<MLXArray>("mtp.positionDeltas")

/// The MTP iterator sets this key on the ``LMOutput/State`` it passes into
/// the main model on each call to opt the target into emitting
/// ``mtpLastHiddenStatesKey`` and ``mtpSharedKVStatesKey``. An absent key
/// reads as `false` (no emit), so non-MTP callers are unaffected.
public let mtpEmitFlagKey = LMOutput.Key<Bool>("mtp.emitDrafterState")

/// Preserve ordinary singleton decode numerics during native MTP verification.
internal let mtpExactVerificationKey = LMOutput.Key<Bool>("mtp.exactVerification")

/// Requests a recurrent-cache checkpoint after this many verification input
/// tokens. Hybrid Qwen models use `1` for MTP-1 so a rejected draft restores
/// state after the always-committed bonus token without replaying the model.
public let mtpCacheCheckpointIndexKey =
    LMOutput.Key<Int>("mtp.cacheCheckpointIndex")

/// Which cache entry each ``mtpSharedKVStatesKey`` tuple was read from, keyed the same way.
///
/// A speculative round keeps only part of what its verify pass wrote, so the emitted snapshot has
/// to be reconciled against the cache afterwards. That reconciliation is exact only if the
/// consumer knows the entry behind each tuple: a sliding layer's snapshot is bounded by its ring
/// and a global layer's is not, and at the crossing the two are indistinguishable by length.
///
/// Public alongside its siblings, because it is part of the same contract rather than an
/// implementation detail of one writer: a target that sets ``mtpSharedKVStatesKey`` is expected to
/// set this too, and a snapshot that arrives without it is refused rather than reconciled against
/// a guessed bound. A target outside this package cannot satisfy that contract without it.
public let mtpSharedKVSourceIndicesKey =
    LMOutput.Key<[String: Int]>("mtp.sharedKVSourceIndices")

/// Errors thrown when a warm KV cache is continued without the model state it requires.
///
/// Models that place tokens with M-RoPE (the Qwen vision families, GLM-OCR) keep a continuation
/// anchor in ``LMOutput/State`` alongside the KV cache. Continuing a warm cache without that
/// anchor positions new tokens as if the cached prefix contained no images, silently changing
/// the output rather than failing.
///
/// Save and restore the pair with `savePromptCache(url:cache:metadata:state:)` and
/// `loadPromptCacheSnapshot(url:)`.
public enum ContinuationStateError: LocalizedError, Equatable {
    /// A cache warmed to a non-zero offset was supplied without the state key the model needs.
    case missingState(model: String, key: String)

    /// A warm cache was supplied for a batched continuation, but this model carries a single
    /// position anchor.
    case unsupportedBatchContinuation(model: String)

    public var errorDescription: String? {
        switch self {
        case .missingState(let model, let key):
            """
            \(model) cannot continue a warm prompt cache without the model state key '\(key)'. \
            Restore the cache with loadPromptCacheSnapshot(url:) and pass the snapshot to \
            ChatSession(_:promptCache:); a file saved without model state must be rebuilt.
            """
        case .unsupportedBatchContinuation(let model):
            """
            \(model) cannot continue a warm prompt cache for more than one batch row. \
            Continue each row with its own cache and state.
            """
        }
    }
}

/// The result of the call to ``LanguageModel/prepare(_:cache:state:prefill:)``
public enum PrepareResult {
    /// tokens to process by the ``AutoregressiveIterator``
    case tokens(LMInput.Text)

    /// logits representing the next token
    case logits(LMOutput)
}

// MARK: - ChatConventionsProviding

/// A model's chat conventions: how it encodes tool calls and how it reasons.
///
/// This knowledge lives with the model definition rather than in a centralized
/// `model_type` table. ``LanguageModel`` conforms with `nil` defaults, so the model
/// factories read `model.toolCallFormat` / `model.reasoningConfig` directly. A model
/// opts in by overriding either property in an extension.
///
/// Callers can override these conventions with ``ModelMetadata`` when loading.
/// Checkpoint-specific defaults, such as DeepSeek-R1 reasoning, are resolved by the loader.
public protocol ChatConventionsProviding {
    /// The tool-call format this model emits, or `nil` for the JSON default.
    var toolCallFormat: ToolCallFormat? { get }

    /// The model's reasoning protocol, or `nil` for non-reasoning models.
    var reasoningConfig: ReasoningConfig? { get }
}

extension ChatConventionsProviding {
    public var toolCallFormat: ToolCallFormat? { nil }
    public var reasoningConfig: ReasoningConfig? { nil }
}

/// Interface for all Language Models (e.g. LLM, VLM).
///
/// The language model is typically called by the ``AutoregressiveIterator`` and it:
///
/// - consumes the ``LMInput``
/// - calls ``prepare(_:cache:state:prefill:)`` to initialize the KVCache and consume the prompt
/// - calls ``callAsFunction(_:cache:state:)-9kuvf`` for each token, producing an ``LMOutput``
/// - the ``AutoregressiveIterator`` accumulates this information into a ``ModelGenerationSummary``
public protocol LanguageModel: BaseLanguageModel, ChatConventionsProviding {

    /// Build derived state after checkpoint or adapter topology updates and
    /// before the model is used for inference.
    ///
    /// Implementations may materialize arrays or replace storage-sharing
    /// module views. The library invokes this lifecycle hook while it has
    /// exclusive access to the model; inference calls must remain read-only.
    func prepare() throws

    /// Prepare the cache state and consume the ``LMInput``.
    ///
    /// `state` is the ``LMOutput/state`` a caller carried over from earlier
    /// evaluation against the same `cache` — present when `cache` is already
    /// warm (a multi-turn chat, a tool-call restart, a restored prompt
    /// cache). Models that keep per-call positional state (e.g. the M-RoPE
    /// `ropeDeltas` of the Qwen VLM families) use it to anchor the new
    /// tokens' positions at the cache offset; models without such state can
    /// ignore it. In the typical cold call it is `nil`.
    ///
    /// This can return:
    /// - ``PrepareResult/tokens(_:)`` if the caller should evaluate the (remaining) tokens normally
    /// - ``PrepareResult/logits(_:)`` to produce the next token from the prompt
    ///
    /// Implementations that chunk the prompt should drive the loop with
    /// ``ModelPrefillOptions/forEachChunk(total:reserving:defaultStepSize:maximumStepSize:_:)``,
    /// which owns cancellation, pooling, and per-chunk progress. An
    /// implementation returning `.logits` owns its whole
    /// ``ModelPrefillOptions/progress`` sequence, including the terminal
    /// `(total, total)`; one returning `.tokens` reports only its own chunks —
    /// the iterator that evaluates the remainder completes the sequence.
    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: ModelPrefillOptions
    )
        throws -> PrepareResult

    /// Primary entry point to produce a step (single token) from the model
    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput

    /// Models may implement this simplified interface if they do not produce any ``LMOutput/State``
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray

    /// Create a new array of ``KVCache`` appropriate for this model.
    ///
    /// Implementations must honor ``KVCacheConfiguration/capacity`` for any
    /// caller-configurable attention layer. Architecture-defined sliding windows
    /// retain their native bounds; recurrent state is unaffected (see
    /// ``makeAttentionKVCache(configuration:)``).
    ///
    /// - Throws: ``KVCacheConfigurationError`` when the request or a model-defined
    ///   cache size is invalid.
    ///
    /// Automatic implementation if self implements ``KVCacheDimensionProvider``.
    func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache]

    /// Authoritative planned status of the cache ``newCache(configuration:)`` produces.
    ///
    /// Use this from ``NativModel`` (or directly on the model)
    /// to inspect topology, requested capacity, and strategy compatibility without
    /// allocating or casting probe caches in application code.
    ///
    /// The default implementation derives the description from ``newCache(configuration:)``.
    /// Models may override with a zero-allocation declarative path, but must stay
    /// consistent with ``newCache(configuration:)``.
    func cacheStatus(configuration: KVCacheConfiguration?) throws -> KVCacheStatus
}

extension LanguageModel {
    /// Most language models have no derived inference state to prepare.
    public func prepare() throws {}

    public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        let logits = callAsFunction(input.tokens, cache: cache)
        return .init(logits: logits)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError("callAsFunction(inputs:cache:) not implemented for \(Self.self)")
    }

    /// Default: classify the caches that ``newCache(configuration:)`` constructs.
    ///
    /// Empty caches allocate negligible state (no tensors until the first update),
    /// so this is always consistent with the runtime path. Prefer a declarative
    /// override only when construction itself is expensive.
    public func cacheStatus(configuration: KVCacheConfiguration? = nil) throws -> KVCacheStatus {
        KVCacheStatus(
            cache: try newCache(configuration: configuration), configuration: configuration,
            phase: .planned)
    }
}

/// Marker protocol for LLMModels
public protocol LLMModel: LanguageModel, LoRAModel {

    /// Models can implement this is they need a custom `MessageGenerator`.
    ///
    /// The default implementation returns `DefaultMessageGenerator`.
    func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator
}

extension LLMModel {

    /// Default prepare step for ``LLMModel``.
    ///
    /// Evaluates the prompt into the cache in chunks of at most
    /// `ModelPrefillOptions.stepSize` (default 256 on iOS/iPadOS, 512 elsewhere), evaluating the final tail with
    /// the continuation state produced by the preceding chunks. With
    /// `ModelPrefillOptions.Chunking.balanced` (the default) the chunks are equal-sized, so no forward is a small
    /// remainder paying full attention cost against the whole prompt.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let stepSize = prefill.resolvedStepSize()
        let y = input.text
        let total = y.tokens.size

        // A prompt that fits in one chunk is handed to the iterator whole,
        // keeping short prompts bitwise-identical to the pre-chunking path.
        // `.unchunked` (forEachChunk processes nothing) takes the same route
        // at any prompt length.
        guard total > stepSize else { return .tokens(y) }

        let inferenceErrors = InferenceErrorCheck(self)
        var processed = 0
        var preparedState = state
        try withPreparedCache(cache, lengths: y.sequenceLengths) {
            // asyncEval lets the CPU build chunk N+1's graph while the GPU evaluates
            // chunk N. Under .remainder the reserved tail is the leftover
            // (up to a full step) rather than a single token.
            processed = try prefill.forEachChunk(
                total: total, reserving: prefill.chunking == .remainder ? stepSize : 1
            ) { range in
                let input = y[.newAxis, range]
                let output = self(input, cache: cache.isEmpty ? nil : cache, state: preparedState)
                preparedState = output.state
                try inferenceErrors.check()
                asyncEval(cache)
            }

            // Single sync after the loop to flush any remaining async work.
            if processed > 0 {
                eval(cache)
            }
        }

        guard processed > 0 else { return .tokens(y) }
        try Task.checkCancellation()
        let tail = y[processed...]
        let result = withPreparedCache(cache, lengths: tail.sequenceLengths) {
            self(tail[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: preparedState)
        }
        try inferenceErrors.check()
        prefill.progress?(total, total)
        return .logits(result)
    }

    public func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator {
        DefaultMessageGenerator()
    }
}

public protocol VLMModel: LanguageModel, LoRAModel {
}
