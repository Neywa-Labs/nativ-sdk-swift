# Ming-Image 0.1 Design

Ming-Image generates posters, interfaces, and other text-rich designs, including images with transparency.

## Model repositories

| Repository | Notes |
| --- | --- |
| [inclusionAI/Ming-Image-0.1-Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design) | Original BF16 components |
| [nativ-community/Ming-Image-0.1-Design-MLX-4bit](https://huggingface.co/nativ-community/Ming-Image-0.1-Design-MLX-4bit) | 4-bit MLX conversion |
| [nativ-community/Ming-Image-0.1-Design-MLX-8bit](https://huggingface.co/nativ-community/Ming-Image-0.1-Design-MLX-8bit) | 8-bit MLX conversion |

## CLI

```sh
swift run nativ image \
  --model nativ-community/Ming-Image-0.1-Design-MLX-4bit \
  --prompt "RGBA, 4-channel, transparent background, a small blue glass icon" \
  --seed 42 \
  --output ./mingimage.png
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativImageModel.loadModel(
    from: .repository("nativ-community/Ming-Image-0.1-Design-MLX-4bit"))
let result = try await model.generate(
    prompt: "RGBA, 4-channel, transparent background, a small blue glass icon",
    parameters: .init(seed: 42)
).result()
try result.image.savePNG(to: URL.documentsDirectory.appending(path: "generated.png"))
```

## Generation options

Use dimensions divisible by 16. The model defaults to 12 steps and requires
guidance 1. It does not support reference-image editing or classifier-free guidance.

The generated image includes an alpha channel. Request transparency explicitly in
the prompt, for example with `RGBA, 4-channel, transparent background`. The CLI
preserves the alpha channel when saving PNG output.

See the [image guide](../../ImageGeneration.md) for results, reference images, and cancellation.
