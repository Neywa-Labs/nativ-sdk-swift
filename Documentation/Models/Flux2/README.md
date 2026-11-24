# FLUX.2 Klein

FLUX.2 Klein generates images from text and edits reference images.

## Checkpoints

Supported checkpoints include Klein 4B, 9B, Base 4B/9B, and 9B KV.

## CLI

```sh
swift run nativ image \
  --model black-forest-labs/FLUX.2-klein-9b-kv \
  --prompt "A crisp red apple on a simple wooden table." \
  --size 256x256 \
  --seed 1234 \
  --output ./flux2.png
```

Image edit:

```sh
swift run nativ image \
  --model black-forest-labs/FLUX.2-klein-9b-kv \
  --task edit \
  --image ./reference.png \
  --prompt "Restyle this image as a watercolor painting." \
  --size 512x512 \
  --output ./flux2-edit.png
```

## Swift

```swift
import Foundation
import NativSDK

let model = try await NativImageModel.loadModel(
    from: .repository("black-forest-labs/FLUX.2-klein-9b-kv"))
let result = try await model.generate(
    prompt: "A crisp red apple on a simple wooden table.",
    parameters: .init(seed: 42, width: 512, height: 512)
).result()
try result.image.savePNG(to: URL.documentsDirectory.appending(path: "generated.png"))
```

Use `.directory(modelDirectory)` to load a local checkpoint. See the
[image generation guide](../../ImageGeneration.md) for checkpoint metadata and editing.

## Notes

- Supported output sizes are 256 through 2048 pixels per side, multiples of 16.
- The 9B KV variant supports reference-image editing.
