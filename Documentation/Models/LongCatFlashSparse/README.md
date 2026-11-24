# LongCat-Flash-Lite-Sparse

LongCat-Flash-Lite-Sparse supports text generation and conversations.

## Model repositories

| Repository | Notes |
| --- | --- |
| [mlx-community/LongCat-Flash-Lite-Sparse-4bit](https://huggingface.co/mlx-community/LongCat-Flash-Lite-Sparse-4bit) | 4-bit MLX checkpoint |
| [mlx-community/LongCat-Flash-Lite-Sparse-6bit](https://huggingface.co/mlx-community/LongCat-Flash-Lite-Sparse-6bit) | 6-bit MLX checkpoint |
| [mlx-community/LongCat-Flash-Lite-Sparse-8bit](https://huggingface.co/mlx-community/LongCat-Flash-Lite-Sparse-8bit) | 8-bit MLX checkpoint |

## CLI

```sh
swift run nativ generate \
  --model mlx-community/LongCat-Flash-Lite-Sparse-4bit \
  --prompt "Explain sparse attention in one paragraph." \
  --max-tokens 256
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(from: .repository("mlx-community/LongCat-Flash-Lite-Sparse-4bit"))
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

Keep the complete checkpoint when copying it to a local directory.
