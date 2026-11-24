# Ternary Bonsai 2

Ternary Bonsai 2 is a 27B model for conversations about text, images, and video, available as a compact 2-bit checkpoint.

## Model repositories

| Repository | Notes |
| --- | --- |
| [prism-ml/Ternary-Bonsai-2-27B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) | Packed MLX checkpoint with the vision encoder |

## CLI

```sh
swift run nativ generate \
  --model prism-ml/Ternary-Bonsai-2-27B-mlx-2bit \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(from: .repository("prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"))
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

The published MLX checkpoint loads directly. Keep all configuration and weight files when copying it locally. This model is separate from the [Bonsai image generator](../Bonsai/README.md).
