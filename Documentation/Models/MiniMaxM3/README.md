# MiniMax M3

MiniMax M3 supports text conversations and still-image prompts.

## Model repositories

| Repository | Notes |
| --- | --- |
| [MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) | Original multimodal checkpoint |

## CLI

```sh
swift run nativ generate \
  --model MiniMaxAI/MiniMax-M3 \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(from: .repository("MiniMaxAI/MiniMax-M3"))
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

Text and still images are supported; video, audio, EAGLE3, and MTP drafting are unsupported for this family. Choose a compatible quantized checkpoint if the original model exceeds available memory.
