# Mistral 3

Mistral 3 supports text and image conversations.

## Model repositories

| Repository | Notes |
| --- | --- |
| [nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit](https://huggingface.co/nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit) | Quantized vision-language checkpoint |

## CLI

```sh
swift run nativ generate \
  --model nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(from: .repository("nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit"))
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

Use a complete multimodal checkpoint for image prompts. See the separate [Small 4](../Mistral4/README.md) and [Large 3](../MistralLarge3/README.md) guides for those models.
