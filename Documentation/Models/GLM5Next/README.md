# GLM-5-Next

GLM-5-Next supports text conversations and still-image prompts.

## Model repositories

| Repository | Notes |
| --- | --- |
| [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | GLM-5-Next checkpoint |

## CLI

```sh
swift run nativ generate \
  --model zai-org/GLM-5.3-Flash \
  --prompt "Explain sparse attention in one paragraph." \
  --max-tokens 256
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(from: .repository("zai-org/GLM-5.3-Flash"))
let input = UserInput(chat: [.user("Explain sparse attention in one paragraph.")])
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

Text and still images are supported. Draft generation is unsupported for this family.
