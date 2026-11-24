# Gemma 4 Unified

Gemma 4 supports image understanding and audio input with compatible checkpoints.

- Example checkpoint: `mlx-community/gemma-4-e2b-it-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/gemma-4-e2b-it-4bit \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

For an audio-capable checkpoint:

```sh
swift run nativ generate \
  --model mlx-community/gemma-4-e2b-it-4bit \
  --audio ./clip.wav \
  --prompt "Transcribe this audio." \
  --max-tokens 256
```

Audio and images can be supplied together, including a spoken question about an image:

```sh
swift run nativ generate \
  --model mlx-community/gemma-4-e2b-it-4bit \
  --image ./image.jpg --audio ./question.wav \
  --prompt "Answer the spoken question using the image." \
  --max-tokens 256
```

Audio is decoded to mono at the checkpoint's sample rate (16 kHz by default).
The feature extractor limits each clip to 30 seconds by default. Checkpoints
without an audio encoder report an error when given audio.

## Swift

```swift
let source = ModelSource.repository("mlx-community/gemma-4-e2b-it-4bit")
let input = UserInput(chat: [
    .user("Describe this image.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

For a spoken question, include audio in the same user message:

```swift
let input = UserInput(chat: [
    .user(
        "Answer the spoken question using the image.",
        images: [.url(URL(fileURLWithPath: "image.jpg"))],
        audios: [.url(URL(fileURLWithPath: "question.wav"))]
    )
])
try await generateText(source: source, input: input)
```

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.

## Draft models

Use `NativDraftModel` for DFlash or DSpark generation with a matching Gemma 4
checkpoint. Text and image prompts, sampling, and `ModelSession` are supported.

| Target | Draft | Default / maximum proposals |
| --- | --- | --- |
| Gemma 4 12B | `deepseek-ai/dspark_gemma4_12b_block7` | 7 / 7 |
| Gemma 4 26B A4B | `z-lab/gemma-4-26B-A4B-it-DFlash` | 7 / 15 |
| Gemma 4 31B | `z-lab/gemma-4-31B-it-DFlash` | 7 / 15 |

```swift
let model = try await NativDraftModel.loadModel(
    target: .repository("lmstudio-community/gemma-4-12B-it-MLX-4bit"),
    draft: .repository("deepseek-ai/dspark_gemma4_12b_block7"))
let session = ModelSession(model: model)
let response = try await session.respond(to: "Explain how rainbows form.")
print(try await response.response())
```

```sh
swift run nativ generate \
  --model mlx-community/gemma-4-26b-a4b-it-4bit \
  --draft-model z-lab/gemma-4-26B-A4B-it-DFlash \
  --image ./image.jpg \
  --prompt "Describe this image."
```

Quantized targets are supported. The draft must match the target's model size;
these drafts cannot be paired with Gemma 4 E2B or E4B. Gemma 4 responses may vary
from ordinary generation or across proposal counts, even with a fixed seed.
Audio and video prompts are unsupported when drafting.

See [speculative generation](../Text/README.md#speculative-generation) for sampling,
proposal counts, and session options.
