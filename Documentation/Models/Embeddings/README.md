# Embeddings

Load a `NativEmbeddingModel` with its static `loadModel` method to encode text or supported media for search and retrieval.

## Encode text

```swift
import NativSDK

let model = try await NativEmbeddingModel.loadModel(
    from: .repository("sentence-transformers/all-MiniLM-L6-v2"))
let result = try await model.embed(
    input: .text("A small observatory on a mountain.")
).result()
if case .vector(let embedding) = result.embeddings {
    print(embedding.count, result.tokenCount)
}
```

Pooled encoders return `.vector([Float])`. Late-interaction encoders such as ColBERT return `.tokenVectors([[Float]])`, with a vector for each input token.

## Retrieval instructions

Include any query or document instructions required by your checkpoint. For example, Qwen3 retrieval queries can specify the task:

```swift
let model = try await NativEmbeddingModel.loadModel(
    from: .repository("Qwen/Qwen3-Embedding-0.6B"))
let query = try await model.embed(
    input: .text("Instruct: Retrieve passages about astronomy\nQuery: Where are observatories built?")
).result()
let document = try await model.embed(
    input: .text("Observatories are often built on mountains with clear skies.")
).result()
```

The checkpoint determines pooling, normalization, and the input length limit. The loader does not add retrieval instructions or silently truncate text.

Visual encoders also accept `.image(url, prompt:)` and, where supported, `.video(url, prompt:)`. See the [embedding guide](../../Embeddings.md) for supported encoders, media examples, configuration, and prepared inputs. The [BERT / MiniLM README](../BERT/README.md) covers the compact encoder used above.

[EmbeddingGemma 2](../EmbeddingGemma2/README.md) adds audio and composed inputs, with normalized Matryoshka prefixes through `dimensions:`.
