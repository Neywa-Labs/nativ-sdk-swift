# K2-Horizon

K2-Horizon is a text model family available in several sizes.

## Model repositories

| Repository | Notes |
| --- | --- |
| [IFM/K2-Horizon-0.9B](https://huggingface.co/IFM/K2-Horizon-0.9B) | Dense text checkpoint |
| [IFM/K2-Horizon-3.7B](https://huggingface.co/IFM/K2-Horizon-3.7B) | Dense text checkpoint |
| [IFM/K2-Horizon-7B](https://huggingface.co/IFM/K2-Horizon-7B) | Dense text checkpoint |
| [IFM/K2-Horizon-32B](https://huggingface.co/IFM/K2-Horizon-32B) | Dense text checkpoint |

## CLI

```sh
swift run nativ generate \
  --model IFM/K2-Horizon-0.9B \
  --prompt "The capital of France is" \
  --max-tokens 64
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(from: .repository("IFM/K2-Horizon-0.9B"))
let input = UserInput(prompt: .text("The capital of France is"))
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

Choose a dense K2-Horizon checkpoint; MoVA and mixture-of-experts K2 variants are unsupported. Use raw text for base checkpoints and chat inputs for instruction checkpoints.
