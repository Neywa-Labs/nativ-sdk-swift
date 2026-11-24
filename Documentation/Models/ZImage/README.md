# Z-Image

Z-Image generates images from text and transforms a single reference image.

## Model repositories

| Repository | Notes |
| --- | --- |
| [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Turbo: 9 steps, guidance 0 |
| [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) | Base: 50 steps, guidance 4 |

## CLI

```sh
swift run nativ image \
  --model Tongyi-MAI/Z-Image-Turbo \
  --prompt "An ink drawing of a small observatory" \
  --seed 42 \
  --output ./zimage.png
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativImageModel.loadModel(
    from: .repository("Tongyi-MAI/Z-Image-Turbo"))
let result = try await model.generate(
    prompt: "An ink drawing of a small observatory",
    parameters: .init(seed: 42)
).result()
try result.image.savePNG(to: URL.documentsDirectory.appending(path: "generated.png"))
```

## Generation options

Use positive dimensions divisible by 16. Turbo disables classifier-free guidance
at guidance values up to 1. The base model supports `negativePrompt` and
`cfgTruncation` (`--negative-prompt` and `--cfg-truncation`).

For image-to-image sampling, add `--task edit --image ./reference.png --strength 0.6`.
Exactly one reference is supported. Strength must be greater than 0 and at most 1.
The Swift API selects this path when `referenceImages` is nonempty.

Original and MLX-converted checkpoints are supported. Keep the scheduler and tokenizer files.

See the [image guide](../../ImageGeneration.md) for results, reference images, and cancellation.
