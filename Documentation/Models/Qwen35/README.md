# Qwen3.5 Vision-Language

Qwen3.5 supports conversations about text, images, and video. Compatible Qwen3.6 checkpoints use the same APIs.

## Model repositories

| Repository | Notes |
| --- | --- |
| [ornith-ai/Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B) | Dense vision-language checkpoint |
| [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) | Mixture-of-experts checkpoint |
| [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | Compatible Qwen3.6 checkpoint |

## CLI

```sh
swift run nativ generate \
  --model ornith-ai/Ornith-1.5-9B \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(from: .repository("ornith-ai/Ornith-1.5-9B"))
let input = UserInput(chat: [.user("Describe this image.", images: [.url(URL(fileURLWithPath: "./image.jpg"))])])
let request = model.generate(
    input: input,
    options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 256)))
for try await event in request {
    if case let .chunk(text) = event {
        print(text, terminator: "")
    }
}
```

## DFlash generation

Use `NativDraftModel` to load the target and its matching DFlash head together:

```swift
let model = try await NativDraftModel.loadModel(
    target: .repository("mlx-community/Qwen3.5-4B-4bit"),
    draft: .repository("z-lab/Qwen3.5-4B-DFlash"))
let input = UserInput(chat: [.user("Describe this image.",
    images: [.url(URL(fileURLWithPath: "./image.jpg"))])])
print(try await model.generate(input: input).response())
```

The same pair accepts text prompts and works with `ModelSession`. Greedy is the
default; temperature, top-p, top-k, min-p and seeds use ordinary generation options.
The CLI exposes the same path:

```sh
swift run nativ generate --model mlx-community/Qwen3.5-4B-4bit \
  --draft-model z-lab/Qwen3.5-4B-DFlash --image ./image.jpg \
  --prompt "Describe this image."
```

Seven proposals is the default; use `speculation: .init(draftTokenCount: 15)` or
`--draft-tokens 15` for the maximum. Metrics report draft acceptance. Use unbounded,
full-precision caches. Batches, penalties, token suppression, video, and audio are
unsupported for this draft pair.

## Notes

Text, image, and video inputs use the same model. Choose a complete multimodal
checkpoint for image and video prompts. For greedy native MTP heads, use
`NativDraftModel`; see the [text guide](../Text/README.md#speculative-generation).
