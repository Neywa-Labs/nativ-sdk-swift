# LLaDA-Image

LLaDA-Image generates images from text and edits a single reference image.

## Model repositories

| Repository | Notes |
| --- | --- |
| [inclusionAI/LLaDA-Image](https://huggingface.co/inclusionAI/LLaDA-Image) | Base BF16 |
| [inclusionAI/LLaDA-Image-Turbo](https://huggingface.co/inclusionAI/LLaDA-Image-Turbo) | Turbo BF16 |
| [inclusionAI/LLaDA-Image-FP8](https://huggingface.co/inclusionAI/LLaDA-Image-FP8) | Base block-FP8 |
| [inclusionAI/LLaDA-Image-Turbo-FP8](https://huggingface.co/inclusionAI/LLaDA-Image-Turbo-FP8) | Turbo block-FP8 |

## CLI

```sh
swift run nativ image \
  --model inclusionAI/LLaDA-Image-Turbo \
  --prompt "A red ceramic teapot on a wooden table" \
  --seed 42 \
  --output ./lladaimage.png
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativImageModel.loadModel(
    from: .repository("inclusionAI/LLaDA-Image-Turbo"))
let result = try await model.generate(
    prompt: "A red ceramic teapot on a wooden table",
    parameters: .init(seed: 42)
).result()
try result.image.savePNG(to: URL.documentsDirectory.appending(path: "generated.png"))
```

## Generation options

Turbo defaults to four steps and guidance 1; the base schedule defaults to 50
steps and guidance 5.
`stochasticSampling` overrides stochastic sampling; the CLI exposes
`--stochastic-sampling` and `--deterministic-sampling`.

Use `useImageTokenPrior: true` (`--image-token-prior`) for optional additional
image conditioning. This increases generation time.

Generation dimensions must be multiples of 16. Editing accepts exactly one
reference and requires dimensions divisible by 32. Use
`--task edit --image ./reference.png`, or pass `referenceImages` to the model.
Negative prompts apply when guidance exceeds 1.

See the [image guide](../../ImageGeneration.md) for results, reference images, and cancellation.
