# Qwen-Image 2.1

Qwen-Image 2.1 generates images and follows editing instructions with one or more references. Editing supports transparency.

## Model repositories

| Repository | Notes |
| --- | --- |
| [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | Original generation and editing checkpoint |

The original repository loads directly.

## CLI

```sh
swift run nativ image \
  --model Qwen/Qwen-Image-2.1 \
  --prompt "A cobalt blue teapot in soft window light" \
  --seed 42 \
  --output ./qwenimage.png
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativImageModel.loadModel(
    from: .repository("Qwen/Qwen-Image-2.1"))
let result = try await model.generate(
    prompt: "A cobalt blue teapot in soft window light",
    parameters: .init(seed: 42)
).result()
try result.image.savePNG(to: URL.documentsDirectory.appending(path: "generated.png"))
```

## Generation options

For editing, add `--task edit --image ./reference.png`; repeat `--image` for more
references. Use `--reference-resolution` to control how much reference-image detail
is retained during editing.
Reference resizing preserves aspect ratio and alpha.

Reference caching reduces repeated processing during an edit and is enabled by
default. Disable it with `useReferenceCache: false`
or `--no-reference-cache`. Set output width and height to your desired size; output
size does not automatically follow the reference aspect ratio.

See the [image guide](../../ImageGeneration.md) for results, reference images, and cancellation.
