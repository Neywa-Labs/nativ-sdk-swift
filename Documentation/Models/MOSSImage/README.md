# MOSS-Image

MOSS-Image generates RGB images from natural-language captions, tag sequences,
or JSON layouts. Pass layout JSON as the prompt string and use the canvas size
for which the layout was authored. [Layout format and examples](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md#layout-prompts-experimental)

| Checkpoint | Description |
| --- | --- |
| [OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) | Base checkpoint |
| [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) | Supervised fine-tuned checkpoint |

The SDK loads the official snapshots and local checkpoints converted with
`mlx-vlm`. Both variants default to 1024 × 1024, 50 steps, guidance 3, and an
empty negative prompt. Width and height must be positive multiples of 16.
Set `maxPromptLength` to allow up to 4096 caption tokens.

## CLI

```sh
swift run nativ image --model OpenMOSS-Team/MOSS-Image-SFT \
  --prompt "A red ceramic teapot on a wooden table in soft morning light" \
  --size 1024x1024 --seed 42 --output teapot.png
```

## Swift

```swift
import Foundation
import NativSDK

let model = try await NativImageModel.loadModel(
    from: .repository("OpenMOSS-Team/MOSS-Image-Base"))
let result = try await model.generate(
    prompt: "A red ceramic teapot on a wooden table in soft morning light",
    parameters: .init(seed: 42, negativePrompt: "text, watermark")
).result()
try result.image.savePNG(to: URL(fileURLWithPath: "teapot.png"))
```

Use `parameters.mossImage` to customize layer skipping. The default negative
guidance pass skips zero-based transformer block 22. Set
`unconditionalSkipLayers` to an empty array to run every block. Set
`skipLayerGuidanceLayers` and a positive `skipLayerGuidanceScale` to enable an
additional positive pass with the selected blocks skipped. Both kinds of
skipping apply from `skipLayerGuidanceStart` (inclusive) through
`skipLayerGuidanceStop` (exclusive), expressed as fractions of the denoising
schedule. Guidance 1 disables the negative pass.

Checkpoints and reference code are available under the
[Apache 2.0 license](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/main/LICENSE).
