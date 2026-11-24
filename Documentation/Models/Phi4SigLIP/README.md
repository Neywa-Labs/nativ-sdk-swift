# Phi-4 Reasoning Vision

Phi-4 Reasoning Vision supports visual reasoning with still images at different resolutions.

## Model repositories

| Repository | Notes |
| --- | --- |
| [microsoft/Phi-4-reasoning-vision-15B](https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B) | 15B vision reasoning checkpoint |

## CLI

```sh
swift run nativ generate \
  --model microsoft/Phi-4-reasoning-vision-15B \
  --image ./image.jpg \
  --prompt "Describe the image and explain your reasoning." \
  --max-tokens 512
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(from: .repository("microsoft/Phi-4-reasoning-vision-15B"))
let input = UserInput(chat: [.user("Describe the image and explain your reasoning.", images: [.url(URL(fileURLWithPath: "./image.jpg"))])])
let request = model.generate(
    input: input,
    options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 512)))
for try await event in request {
    if case let .chunk(text) = event {
        print(text, terminator: "")
    }
}
```

## Notes

Still images are supported; video and audio are unsupported. For audio inputs, see [Phi-4 Multimodal](../Phi4Multimodal/README.md).
