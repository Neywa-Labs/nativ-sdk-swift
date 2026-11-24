# Language models

Text checkpoints use the same `NativModel.loadModel` and `ModelSession` APIs as multimodal models. They need tokenizer metadata and model weights; no image processor configuration is required.

## Chat with conversation history

```swift
import NativSDK

let model = try await NativModel.loadModel(
    from: .repository("nativ-community/LFM3-Omni-2B-4bit")
)
let session = ModelSession(model: model, instructions: "Give clear, concise answers.")
let response = try await session.respond(to: "Explain why the sky is blue.")
print(try await response.response())
let followup = try await session.respond(to: "Why does it look red at sunset?")
print(try await followup.response())
await session.close()
```

Submit only the new prompt each turn; the session retains conversation history and reuses its prompt cache.

## Generation options and metrics

```swift
let request = model.generate(
    input: UserInput(prompt: "Explain gravity briefly."),
    options: .init(
        sampling: .init(temperature: 0.2, topP: 0.9),
        output: .init(maxTokens: 256)))
print(try await request.response())
let metrics = try await request.metrics()
print(metrics.generatedTokenCount, metrics.stopReason)
```

Use `sampling` for temperature, top-p/top-k, penalties, and a seed. `output.maxTokens` defaults to 8192 generated tokens; override it for a different limit, or explicitly pass `nil` for unbounded autoregressive output. Generation can stop earlier on a stopping token or cancellation. Checkpoint metadata supplies stopping and chat conventions.

## Stream a response

```swift
let request = model.generate(input: UserInput(prompt: "Write a short story."))
for try await event in request {
    if case .chunk(let text) = event {
        print(text, terminator: "")
    }
}
```

Choose either `response()` or iteration for each request. Iteration also exposes reasoning and tool-call events. See the [runtime guide](../../Runtime.md) for media inputs, tools, cancellation, cache options, and session lifetimes, and the [model catalog](../README.md) for multimodal families.

## Supported text families

Choose a compatible checkpoint from these families:

| Family | Models |
| --- | --- |
| Llama and Mistral | Llama, Llama 4, Mistral, Mixtral, Mistral 3/4, Ministral 3, IQuestCoder, IQuestLoopCoder |
| Gemma | Gemma, Gemma 2/3/3n/4, RecurrentGemma |
| Qwen | Qwen, Qwen2, Qwen2 MoE, Qwen3, Qwen3 MoE/Next, Qwen3.5 dense and MoE, AceReason |
| Phi | Phi, Phi-3, Phi-3 Small, Phi MoE, Phixtral |
| DeepSeek | DeepSeek, V2, V3, V3.2, V4, Kimi K2, JoyAI LLM Flash |
| GLM | GLM, GLM4, GLM4 MoE/Lite, GLM MoE DSA, Solar Open |
| Granite | Granite, Granite MoE, Granite MoE Hybrid |
| Nemotron and recurrent models | Nemotron, Nemotron NAS/H, Falcon H1/Mamba, Jamba, Mamba/Mamba2, Kimi Linear, RWKV7 |
| Liquid | LFM2, LFM2 MoE |
| Other MoE models | AFMoE, Bailing MoE/Linear, Cohere2 MoE, DBRX, Dots1, ERNIE 4.5 MoE, [ERNIE 5.5 Flash Base](../Ernie55/README.md), EXAONE MoE, Flint, Hunyuan, Klear, GPT-OSS, MiniMax/M2/M3, Mellum, MiMo V2 Flash, HyV4, Inkling, LongCat Flash/Ngram/Sparse, Step 3.5, OLMoE |
| Other text models | AFM7, Apertus, Baichuan M1, BitNet, Cohere/Cohere2, ERNIE 4.5, EXAONE/EXAONE4, GPT-2, GPT BigCode, GPT-NeoX, Helium, Hunyuan V1 Dense, InternLM2/3, K2-Horizon, Laguna, Lille-130M, MiniCPM/3, MiMo, Nanbeige, NanoChat, OLMo/2/3, OpenELM, PLaMo/2, Limite, Z1T, Seed-OSS, SmolLM3, StableLM, TeleChat3, StarCoder2, Youtu LLM, HRM Text, Spark-X2.5 |
| Diffusion models | LLaDA 2, Diffusion Gemma, Nemotron Labs Diffusion |

See the [model catalog](../README.md) for checkpoint IDs, examples, and
family-specific input requirements.

Text checkpoints need their weights, tokenizer, configuration, and chat template.
Use raw text for base completion models and chat inputs for instruction models.

## Speculative generation

Load a compatible target and draft together with `NativDraftModel`. The resulting
model supports the same response and streaming APIs:

```swift
let model = try await NativDraftModel.loadModel(
    target: .repository("LiquidAI/LFM2.5-2.6B"),
    draft: .repository("LiquidAI/LFM2.5-2.6B-DSpark"))
print(try await model.generate(
    input: UserInput(prompt: "Write a short Python function.")
).response())
```

| Target | Draft | Inputs | Default / maximum proposals |
| --- | --- | --- | --- |
| Qwen3 or Mistral4 | Smaller independent model with matching tokenizer | Text | 4 / 16 |
| `LiquidAI/LFM2.5-2.6B` | `LiquidAI/LFM2.5-2.6B-DSpark` | Text | 7 / 9 |
| `LiquidAI/LFM2.5-VL-3B` | `LiquidAI/LFM2.5-VL-3B-DSpark` | Text and images | 7 / 9 |
| Dense Qwen3, including quantized targets | Matching DFlash or DSpark | Text | 7 / 15 for block-16 DFlash; 7 / 7 for block-7 DSpark |
| Qwen3.5 | Matching DFlash | Text and images | 7 / 15 |
| Qwen3.5-family | Matching MTP head | Text | 1 / 1 |
| Laguna XS 2.1 / S 2.1 | Matching Laguna DFlash | Text | 15 / 15 |
| Gemma 4 | Matching DFlash or DSpark | Text and images | 7 / 15 for block-16 DFlash; 7 / 7 for block-7 DSpark |

For Qwen3-4B, use `mlx-community/Qwen3-4B-4bit` with
`z-lab/Qwen3-4B-DFlash-b16` or `deepseek-ai/dspark_qwen3_4b_block7`.
For Qwen3.5-4B, use `mlx-community/Qwen3.5-4B-4bit` with
`z-lab/Qwen3.5-4B-DFlash`. See the [Qwen3.5 guide](../Qwen35/README.md#dflash-generation)
for an image example.
See the [Gemma 4 guide](../Gemma4Unified/README.md#draft-models) for compatible
Gemma 4 checkpoints, and the [Laguna guide](../Laguna/README.md#draft-models) for
Laguna XS 2.1 and S 2.1 pairs.

Choose proposal count with `speculation: .init(draftTokenCount: 4)`, or omit it for
the model default. Greedy sampling is the default. Independent drafts and MTP
require greedy sampling; DFlash and DSpark also support temperature, top-p, top-k,
min-p, and seeds:

```swift
let request = model.generate(
    input: UserInput(prompt: "Write a short story."),
    options: .init(sampling: .init(temperature: 0.7, topP: 0.9, seed: 42)))
print(try await request.response())
```

With the same seed and sampling settings, LFM2 and Qwen DFlash/DSpark produce the
same response as generation without a draft, even when changing the proposal count.
Gemma 4 and Laguna responses may vary with drafting or proposal count, even with a
fixed seed. Performance depends on the pair, prompt, and sampling settings;
drafting may be slower for some requests.

LFM2 DSpark requires the original unquantized BF16 dense target. Quantized LFM2
and LFM2-MoE targets are unsupported. DFlash and DSpark require unbounded,
full-precision caches. Batches, penalties, token suppression, video, and audio
are unsupported for these draft paths. DFlash2 and EAGLE3 are unsupported.

For native MTP, load a matching head explicitly:

```swift
let model = try await NativDraftModel.loadModel(
    target: .repository("mlx-community/Qwen3.6-27B-mxfp4"),
    draft: .repository("mlx-community/Qwen3.6-27B-MTP-4bit"))
print(try await model.generate(input: UserInput(prompt: "Explain binary search.")).response())
```

The draft must match the target family and size. Omit `speculation` for the
one-proposal MTP default. Metrics identify the strategy and report accepted and
proposed tokens. The [CLI](../../../Sources/Tools/nativ/README.md#draft-models)
provides the same loading options through `--draft-model`.

## Conversations with a draft model

Use the loaded pair with `ModelSession`:

```swift
let session = ModelSession(model: model)
let response = try await session.respond(to: "Explain why the sky is blue.")
print(try await response.response())
let followup = try await session.respond(to: "Why does it look red at sunset?")
print(try await followup.response())
await session.close()
```

Submit only new messages. Sessions retain history and tool calls; failed or
cancelled turns leave the previous conversation intact. Text turns reuse their
conversation cache. Use `cacheReuse:` to configure reuse or `clearCache()` to
release it while preserving messages.

Use `generate(input:)` for supported image prompts:

```swift
let model = try await NativDraftModel.loadModel(
    target: .repository("LiquidAI/LFM2.5-VL-3B"),
    draft: .repository("LiquidAI/LFM2.5-VL-3B-DSpark"))
print(try await model.generate(input: UserInput(chat: [
    .user("Describe this image.", images: [.url(imageURL)])
])).response())
```

## Cache options

Default cache settings work across the supported models. Custom capacity and
compression support varies by family. `model.cacheStatus(configuration:)` lets
you check a policy before generating; `allowPartial` applies compression only
where supported. Phi3Small requires unbounded history. See the
[runtime guide](../../Runtime.md#memory-and-conversation-caches) for session memory controls.

## Reranking

Jina Qwen3 ranking checkpoints can score a query against several documents.
Use `ListwiseRerankerModel.score(input:documentCount:)` through `model.withModel`.
Supply a tokenized listwise prompt in `RerankerInput` with one query marker and
one marker per document, using the checkpoint's tokenizer and marker IDs.
Scores are cosine similarities in document order.
