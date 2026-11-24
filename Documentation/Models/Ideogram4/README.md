# Ideogram 4

Ideogram 4 generates images from text prompts.

## Checkpoints

Supported checkpoint: `ideogram-ai/ideogram-4-fp8` (generation only).

## CLI

```sh
swift run nativ image \
  --model ideogram-ai/ideogram-4-fp8 \
  --prompt "A large centered red apple on a simple wooden table." \
  --size 512x512 \
  --seed 42 \
  --output ./ideogram4.png
```

## Swift

```swift
import Foundation
import NativSDK

let model = try await NativImageModel.loadModel(
    from: .repository("ideogram-ai/ideogram-4-fp8"))
let result = try await model.generate(
    prompt: "A large centered red apple on a simple wooden table.",
    parameters: .init(seed: 42, width: 512, height: 512)
).result()
try result.image.savePNG(to: URL.documentsDirectory.appending(path: "generated.png"))
```

Use `.directory(modelDirectory)` to load a local checkpoint. See the
[image generation guide](../../ImageGeneration.md) for checkpoint metadata and editing.

## Notes

- Supported output sizes are 256 through 2048 pixels per side, multiples of 16,
  with an aspect ratio of at most 6:1.
- `--steps 4` selects the default 20-step Ideogram sampler preset. Pass another
  `--steps` value to run that exact number of steps.
- Plain text prompts work directly; structured JSON captions are also supported.
- Enable optional prompt expansion with `parameters.promptExpansion = .model(source)`
  or `--prompt-expansion-model <repo|path>`. Supply a compatible language model.
