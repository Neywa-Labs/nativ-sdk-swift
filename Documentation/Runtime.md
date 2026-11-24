# Using Nativ SDK

Load a model once, then use it for as many requests as you need. Start with the
[quick starts](../README.md), or choose a guide for
[language models](Models/Text/README.md), [images](Models/ImageGeneration/README.md),
[embeddings](Models/Embeddings/README.md), or [decisions](Models/Decisions/README.md).

## Loading models

| Type | Use |
| --- | --- |
| `NativModel` | Text and multimodal responses, conversations, supported audio output |
| `NativImageModel` | Image generation and editing |
| `NativEmbeddingModel` | Embeddings for search and retrieval |
| `NativDecisionModel` | Named choice, score, and Boolean answers |
| `NativDraftModel` | Generation with a compatible target and draft model |

```swift
import NativSDK

let model = try await NativModel.loadModel(
    from: .repository("nativ-community/LFM3-Omni-2B-4bit"))
```

Models download from Hugging Face and are cached for later use. Use
`.repository(id, revision:)` to select a revision, or `.directory(url)` to load
local files. Keep the checkpoint's configuration, tokenizer, and any required
processor files alongside its weights. See the [model catalog](Models/README.md)
for supported checkpoints and preparation requirements.

For MoE checkpoints, pass `expertStreaming: .init(cacheSizeBytes: 64 * 1024 * 1024)`
to stream routed expert weights from disk. The guide covers checkpoint layouts
and memory requirements. [Expert streaming](ExpertStreaming.md)

Use `tokenizerSource:` when the tokenizer is distributed separately. Moondream
uses `moondream/starmie-v1` by default; that tokenizer may download even when the
model is local. For offline use, supply all required local directories through
`ResolvedModelAssets` and call `loadModel(assets:)`.

For custom authentication, download caching, or Hub endpoints, resolve assets with
`ModelArtifactResolver.resolve(downloader:)` and `HuggingFaceDownloader(client:)`,
then pass them to `loadModel(assets:)`.

## Responses and streaming

For a complete textual response:

```swift
let request = model.generate(input: UserInput(prompt: "Why is the sky blue?"))
print(try await request.response())
let metrics = try await request.metrics()
print(metrics.generatedTokenCount, metrics.stopReason)
```

For incremental output:

```swift
let request = model.generate(input: UserInput(prompt: "Write a short story."))
for try await event in request {
    if case .chunk(let text) = event {
        print(text, terminator: "")
    }
}
let metrics = try await request.metrics()
```

Choose `response()` or iteration once per request. `response()` returns response
text; use iteration to receive reasoning, tool calls, and diagnostics as well.
Read the response before awaiting `metrics()`; metrics alone do not read the response.
Image, embedding, decision, and audio operations return their data through `result()`.
Generated audio exposes an evaluated, immutable MLX tensor at `result.waveform`.
Use it directly in MLX or pass it to another task. `result.samples` provides
Swift `[Float]` on first access, and `result.sampleRate` is in Hz. To decode
a local audio file once and reuse it across tasks or models, keep its samples
and source rate together in a waveform.

```swift
let clip = try UserInput.AudioWaveform(contentsOf: url)
let input = UserInput(prompt: "Transcribe this.", audios: [.waveform(clip)])
```

For samples already in memory, use `.array(samples, sampleRate: sourceRate)`.
Nativ snapshots the mono waveform and resamples it when a processor requests a
different rate. For example, `.array(result.waveform, sampleRate: result.sampleRate)`
passes generated audio directly to another model. Use `.array(samples)` for
input already prepared at the receiving model's rate.

Keep the request handle to call `cancel()`. Errors and cancellation throw when
reading the result. Breaking out of a generation loop also cancels the request.
Read streamed events promptly; an unread stream can fail with
`outputBufferOverflow`. `response()` throws on failure rather than returning
partial text.

## Conversations

```swift
let session = ModelSession(model: model, instructions: "Keep answers concise.")
let first = try await session.respond(to: "What causes rain?")
print(try await first.response())
let followup = try await session.respond(to: "How does snow differ?")
print(try await followup.response())
await session.close()
```

Submit only new messages each turn. Use `generate(input:)` for structured messages,
media, and tool results. Sessions retain conversation history; successful turns
are saved, while failed or cancelled turns leave the previous conversation intact.
Treat streamed text as provisional until the request completes.

Use `options:` to override one turn, or `setOptions` to change session defaults.
`withTranscript` reads saved messages, and `replaceTranscript` replaces them.
`clearCache()` frees conversation cache memory while keeping messages. `reset()`
starts a new conversation with the same instructions and defaults. `close()` stops
pending requests and releases the session's conversation and cache.

A session can also use a `NativDraftModel`; see
[speculative generation](Models/Text/README.md#speculative-generation).

## Media and tools

Pass media as attachments to a message:

```swift
let request = model.generate(input: UserInput(chat: [
    .user("Describe this image.", images: [.url(imageURL)])
]))
print(try await request.response())
```

Supported input types vary by checkpoint. Check its family guide before supplying
images, video, or audio. For speech output, see [Qwen3 Omni](Models/Qwen3Omni/README.md).

Provide tool declarations with the input and consume tool-call events from the
response. Your application executes the tools and sends their results back in the
next turn. Nativ SDK never executes tools automatically. Strict validation rejects
arguments that violate supported schema rules; permissive validation lets the
application handle them. See the [Playground example](../Examples/Playground/README.md) for a
working tool conversation.

Reasoning appears separately from response text. Use
`output.separateReasoning = false` to include ordinary reasoning tags in text;
Harmony and Onyx formats always keep reasoning separate.

## Generation options

Use `ModelGenerationOptions.sampling` for temperature, top-p, top-k, min-p,
penalties, and a seed. `output.maxTokens` defaults to 8192 generated tokens.
Set a different limit as needed, or explicitly pass `nil` for unbounded autoregressive
output. Generation can stop earlier on a stopping token or cancellation.
Checkpoints supply their normal stopping rules; `ModelMetadata` lets you add
stops or customize tool and reasoning conventions when loading.

Set `output.format = .tokens` to receive token IDs instead of text.
`sampling.suppressedTokenIDs` excludes selected tokens. These options are supported
for ordinary autoregressive generation; draft and diffusion models have their
own restrictions.

Diffusion models use their checkpoint defaults with `.automatic`, or can be
selected explicitly with `.diffusion`. Use `options.diffusion.maxTokens` to override
their output limit. Diffusion generation does not support tools, raw-token output,
penalties, token suppression, or custom cache/prefill settings. Media support
varies by model.

`generateBatch(inputs:options:)` creates independent requests. Use `responses()` to
consume their streams concurrently and collect a `Result<String, Error>` per input,
in input order:

```swift
let batch = model.generateBatch(inputs: [
    UserInput(prompt: "Explain gravity briefly."),
    UserInput(prompt: "Why is the sky blue?")
])
let responses = await batch.responses()
for (index, response) in responses.enumerated() {
    switch response {
    case .success(let text): print("Response \(index): \(text)")
    case .failure(let error): print("Response \(index) failed: \(error.localizedDescription)")
    }
}
let metrics = await batch.metrics()
```

A row failure or individual cancellation does not cancel other rows. Cancelling
the task awaiting `responses()` cancels all requests and waits for them to finish;
already completed rows may still succeed. Collection follows `response()` semantics:
only text chunks are returned, and raw-token output fails for that row.

For streaming, consume every entry in `batch.requests` concurrently instead of
calling `responses()`. Each request allows only one consumer. Begin consuming promptly:
requests start eagerly, and sequential consumption can overflow an unread stream's
buffer. Await `metrics()` after reading the responses; it does not read streams.
Requests run one at a time, so submitting a batch does not speed up generation.

## Memory and conversation caches

Sessions reuse supported text prompts automatically. Each session has its own
conversation cache, even when several sessions share a loaded model. Configure
reuse and checkpoint limits with `PromptCacheReuseConfiguration`, or disable reuse
to reduce retained memory. `metrics().cache` reports reuse for supported turns.
Media and diffusion turns do not use text-prefix reuse.

For direct generation, `ModelGenerationOptions.cache` controls cache capacity and
compression. Support varies by model: strict compatibility rejects unsupported
settings, while `allowPartial` applies compression where available. Use
`model.cacheStatus(configuration:)` to check a configuration before generating.
Draft model cache requirements are listed in the
[language guide](Models/Text/README.md#speculative-generation).

## Adapters and prepared inputs

See [adapters](Adapters.md) to load, unload, or merge a compatible LoRA/DoRA adapter.

For reusable input preparation workflows, call `prepare(input:)`, then pass the
prepared value to the same model's `generate(prepared:options:)` or
`embed(prepared:)`. A prepared input can be consumed only once.

Use `.materialized()` inside `inspect` or `withModel` to share a tensor snapshot
across tasks. Import `MLX` when working with the tensor directly.

```swift
let tokens = prepared.inspect { $0.text.tokens.materialized() }
```

For video preprocessing, call `.materialized()` on the sequence returned by
`MediaProcessing.asProcessedSequence` to share its frames, timestamps and duration.

For advanced integrations, `withModel` provides scoped access to the loaded model.
Finish work inside that closure and do not retain model resources outside it.
After manually changing weights, tokenizer, or preprocessing settings, clear any
session caches before continuing.
