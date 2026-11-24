# Qwen3.8-Flash-Next

Qwen3.8-Flash-Next supports text conversations and visual understanding.

## Model repositories

| Repository | Notes |
| --- | --- |
| [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | Large multimodal checkpoint |

## CLI

```sh
swift run nativ generate \
  --model Qwen/Qwen3.8-Flash-Next \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(from: .repository("Qwen/Qwen3.8-Flash-Next"))
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

Use a complete checkpoint that includes all embeddings; external `ple_storage` files are unsupported. Native MTP drafting is unsupported for this family. Choose a compatible quantized checkpoint if the original weights exceed available memory.
