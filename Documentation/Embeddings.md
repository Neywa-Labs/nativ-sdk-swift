# Embeddings

Create vectors for search, retrieval, and similarity with `NativEmbeddingModel`.

```swift
import NativSDK

let model = try await NativEmbeddingModel.loadModel(
    from: .repository("Qwen/Qwen3-Embedding-0.6B"))
let result = try await model.embed(input: .text("A mountain observatory")).result()
if case .vector(let values) = result.embeddings {
    print(values.count, result.tokenCount)
}
```

Use `.directory(url)` for local checkpoints. See the
[loading guide](Runtime.md#loading-models) for custom download settings and offline use.

## Supported models

| Family | Result | Media |
| --- | --- | --- |
| [BERT / MiniLM](Models/BERT/README.md) | Vector | Text |
| XLM-RoBERTa | Vector | Text |
| ModernBERT | Vector | Text |
| SigLIP text tower | Vector | Text |
| Bidirectional Llama | Vector | Text |
| Gemma3 / EmbeddingGemma | Vector | Text |
| [EmbeddingGemma 2](Models/EmbeddingGemma2/README.md) | Vector | Text, images, audio, video, composed inputs |
| LFM2 | Vector | Text |
| LFM2 ColBERT | Token vectors | Text |
| Ministral3 | Vector | Text |
| Qwen3 | Vector | Text |
| Qwen3-VL | Vector | Text, images, video |
| ColQwen2.5 | Token vectors | Text, images, video |
| ColIdefics3 | Token vectors | Text, images |
| Llama Nemotron VL | Vector | Text, images |

Compatible MLX-quantized checkpoints are supported. Keep the complete checkpoint,
including any pooling and projection configuration directories.

## Retrieval inputs

Include the checkpoint's query or document instructions in your input, such as
`query: ...`, `passage: ...`, or a Qwen retrieval instruction. The SDK does not add
these instructions automatically. See the [retrieval example](Models/Embeddings/README.md#retrieval-instructions).

For visual encoders, pass a file URL and optional text:

```swift
let imageResult = try await model.embed(
    input: .image(imageURL, prompt: "Describe this document")
).result()
let videoResult = try await model.embed(input: .video(videoURL)).result()
```

EmbeddingGemma 2 also accepts `.audio(url, prompt:)` and
`.multimodal(prompt:images:videos:audios:)`. Request a normalized Matryoshka prefix
with `dimensions: 128`, `256`, or `512`; the default has 768 dimensions. See its
[model guide](Models/EmbeddingGemma2/README.md) for retrieval and media examples.

Choose a model that supports the supplied media type. SigLIP supports text only.
Each operation accepts one input. Empty text without media, tools, unsupported
media, and text beyond the checkpoint's input limit are rejected. Structured
messages are supported by EmbeddingGemma 2 through prepared inputs. Text is not
silently truncated.

## Results

Use `result.tensor` directly in MLX or pass it to another task. This immutable
`MaterializedArray` has shape `[dimensions]` for pooled embeddings and
`[tokenCount, dimensions]` for token embeddings. Masked rows are removed in token
order. For Swift collections, read `result.embeddings` to get one of these forms.

- `.vector([Float])` contains one vector for the input, suitable for vector similarity.
- `.tokenVectors([[Float]])` contains per-token vectors for late-interaction retrieval,
  such as ColBERT. Use a retrieval method designed for these representations.

The checkpoint determines vector size, pooling, and normalization. `tokenCount`
reports input tokens. Keep the operation handle to cancel a request, and await
`result()` to obtain its embeddings.

For images held in memory, use `UserInput.images` with `prepare(input:)`, then
`embed(prepared:)` on the same model. See [prepared inputs](Runtime.md#adapters-and-prepared-inputs).
