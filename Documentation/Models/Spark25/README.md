# Spark-X2.5

Spark-X2.5 is available in instruction and base variants for chat and text completion.

## Model repositories

| Repository | Notes |
| --- | --- |
| [XHToken/Spark-X2.5-4B](https://huggingface.co/XHToken/Spark-X2.5-4B) | Instruction checkpoint |
| [XHToken/Spark-X2.5-4B-Base](https://huggingface.co/XHToken/Spark-X2.5-4B-Base) | Base completion checkpoint |
| [XHToken/Spark-X2.5-1.7B](https://huggingface.co/XHToken/Spark-X2.5-1.7B) | Instruction checkpoint |
| [XHToken/Spark-X2.5-1.7B-Base](https://huggingface.co/XHToken/Spark-X2.5-1.7B-Base) | Base completion checkpoint |

## CLI

```sh
swift run nativ generate \
  --model XHToken/Spark-X2.5-1.7B \
  --prompt "Explain why the sky is blue." \
  --max-tokens 256
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(from: .repository("XHToken/Spark-X2.5-1.7B"))
let input = UserInput(chat: [.user("Explain why the sky is blue.")])
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

Use raw text for base checkpoints and chat inputs for instruction checkpoints.
