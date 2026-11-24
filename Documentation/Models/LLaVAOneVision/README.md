# LLaVA-OneVision

LLaVA-OneVision supports conversations about one or more still images.

## Model repositories

| Repository | Notes |
| --- | --- |
| [llava-hf/llava-onevision-qwen2-0.5b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) | Compact general checkpoint |
| [llava-hf/llava-onevision-qwen2-0.5b-si-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-si-hf) | Single-image tuned |
| [llava-hf/llava-onevision-qwen2-7b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf) | General checkpoint |
| [llava-hf/llava-onevision-qwen2-7b-si-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-si-hf) | Single-image tuned |
| [llava-hf/llava-onevision-qwen2-7b-ov-chat-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-chat-hf) | Chat tuned |
| [llava-hf/llava-onevision-qwen2-72b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-72b-ov-hf) | Large checkpoint; substantial memory required |

## CLI

```sh
swift run nativ generate \
  --model llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(from: .repository("llava-hf/llava-onevision-qwen2-0.5b-ov-hf"))
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

## Notes

Still images, including multiple images, are supported. Video and audio are unsupported. Large images use more of the context window. LLaVA-OneVision 1.5 checkpoints are unsupported.
