# Z1T-0

Z1T-0 is a small research model for short text completions.

## Model repositories

| Repository | Notes |
| --- | --- |
| [AlazarM/Z1T-0-mlx](https://huggingface.co/AlazarM/Z1T-0-mlx) | MLX conversion with FP32 weights |

## CLI

```sh
swift run nativ generate \
  --model AlazarM/Z1T-0-mlx \
  --prompt "The meaning of life is" \
  --max-tokens 64
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(from: .repository("AlazarM/Z1T-0-mlx"))
let input = UserInput(prompt: .text("The meaning of life is"))
let request = model.generate(
    input: input,
    options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 64)))
for try await event in request {
    if case let .chunk(text) = event {
        print(text, terminator: "")
    }
}
```

## Notes

Use the converted MLX repository. The original `Extropic-AI/Z1T-0` checkpoint cannot be loaded directly. This model is suited to short raw-text completions.
