# EmbeddingGemma 2

`google/embeddinggemma-2` encodes text, images, audio, video, and composed inputs into a shared 768-dimensional retrieval space. The SDK loads the original checkpoint and compatible MLX conversions through `NativEmbeddingModel`.

```swift
import NativSDK

let model = try await NativEmbeddingModel.loadModel(
    from: .repository("google/embeddinggemma-2"))
let query = try await model.embed(
    input: .text("task: search result | query: Which planet is known as the Red Planet?"),
    dimensions: 256
).result()
let document = try await model.embed(
    input: .text("title: none | text: Mars is known as the Red Planet."),
    dimensions: 256
).result()
```

The default result is a normalized float32 `.vector` with 768 values. Matryoshka prefixes of 128, 256, or 512 dimensions are renormalized when requested with `dimensions:`. Use the same dimension for queries and documents. Retrieval instructions are supplied explicitly; prompts and media tokens participate in mean pooling, while padding is excluded.

```swift
let image = try await model.embed(input: .image(imageURL)).result()
let audio = try await model.embed(input: .audio(audioURL)).result()
let video = try await model.embed(input: .video(videoURL)).result()
let composed = try await model.embed(
    input: .multimodal(
        prompt: "title: none | text: <|image|> accompanied by <|audio|>",
        images: [imageURL], audios: [audioURL])
).result()
```

Media placeholders are inserted when the prompt has none, in image, video, audio order before the text. Include `<|image|>`, `<|video|>`, and `<|audio|>` yourself to control interleaving. Each placeholder must have a corresponding media input. For in-memory media and structured messages, use `prepare(input:)` with `UserInput`, then `embed(prepared:dimensions:)` on the same model.

The processor restores nested image, audio, and video settings from the checkpoint. Images and video frames use aspect-preserving Gemma 4 patch grids. Video sampling supports the saved frame rate, maximum frame count, uniform or truncate overflow strategy, and timestamps. `UserInput.processing.video` can override the sampling rate or request a frame count. Audio uses the Gemma 4 mel extractor; inputs exceeding its configured sample limit (30 seconds by default) are rejected instead of silently truncated. Text and expanded media inputs exceeding the checkpoint's token limit are also rejected.

To load a reduced local checkpoint, set `vision_config` and/or `audio_config`
to `null` in its `config.json`. Pass media only for enabled inputs. The SDK
also supports the text-only `embedding_gemma2_text` configuration.
