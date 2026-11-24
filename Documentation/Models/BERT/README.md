# BERT / MiniLM embeddings

BERT and MiniLM create sentence embeddings for search and similarity.

## Model repositories

| Repository | Notes |
| --- | --- |
| [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Compact BERT-architecture sentence embeddings |

Load this model with `NativEmbeddingModel.loadModel`.

## CLI

The `nativ` CLI currently has no embedding command. Use the Swift API below;
`generate` is a token-generation command and does not return embeddings.

## Swift API

```swift
import NativSDK

let model = try await NativEmbeddingModel.loadModel(
    from: .repository("sentence-transformers/all-MiniLM-L6-v2"))
let result = try await model.embed(
    input: .text("A small observatory on a mountain")
).result()
if case .vector(let values) = result.embeddings {
    print(values.count, result.tokenCount)
}
```

## Notes

Inputs are plain text. Empty text and inputs beyond the checkpoint’s token limit are rejected. BERT sentence embeddings are supported; reranking, token classification, and redaction are not. See the [embedding guide](../../Embeddings.md) for other encoders.
