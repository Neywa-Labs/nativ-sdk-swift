# Prism Bonsai

Prism Bonsai generates images from text prompts.

## Checkpoints

Supported checkpoint: `prism-ml/bonsai-image-ternary-4B-mlx-2bit` (generation only).

## CLI

```sh
swift run nativ image \
  --model prism-ml/bonsai-image-ternary-4B-mlx-2bit \
  --prompt "A tiny glass bonsai tree on a moonlit desk." \
  --size 512x512 \
  --seed 9909 \
  --output ./bonsai.png
```

## Swift

```swift
import Foundation
import NativSDK

let model = try await NativImageModel.loadModel(
    from: .repository("prism-ml/bonsai-image-ternary-4B-mlx-2bit"))
let result = try await model.generate(
    prompt: "A tiny glass bonsai tree on a moonlit desk.",
    parameters: .init(seed: 42, width: 512, height: 512)
).result()
try result.image.savePNG(to: URL.documentsDirectory.appending(path: "generated.png"))
```

Use `.directory(modelDirectory)` to load a local checkpoint. See the
[image generation guide](../../ImageGeneration.md) for checkpoint metadata and editing.

## Notes

- Supported output sizes are 256 through 2048 pixels per side, multiples of 16.
- Image editing and 1-bit Bonsai variants are not exposed yet.
