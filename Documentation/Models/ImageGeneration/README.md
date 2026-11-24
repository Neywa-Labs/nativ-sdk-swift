# Image generation

Generate images with `NativImageModel`. Each model supplies default sampling settings.

## Generate an image

```swift
import Foundation
import NativSDK

let model = try await NativImageModel.loadModel(
    from: .repository("black-forest-labs/FLUX.2-klein-4B"))
let result = try await model.generate(
    prompt: "A watercolor painting of a mountain observatory.",
    parameters: .init(seed: 42, width: 512, height: 512)
).result()
try result.image.savePNG(to: URL.documentsDirectory.appending(path: "generated.png"))
print(result.width, result.height, result.seed)
```

Use `result.image` to display or save the image. `result` also includes the seed,
dimensions, sampling steps, and guidance used to generate it.

## Display and export

Create a `CGImage` for display or get encoded PNG bytes to upload, share, or store:

```swift
let cgImage = try result.image.cgImage()
let png = try result.image.pngData() // Foundation.Data
```

In SwiftUI, pass the `CGImage` to `Image(decorative: cgImage, scale: 1)`.
UIKit can use `UIImage(cgImage: cgImage)`; AppKit can use
`NSImage(cgImage: cgImage, size: .zero)`. Import the corresponding UI framework.

`savePNG(to:)` creates missing parent directories and atomically replaces an existing
file. It requires a local file URL and always writes PNG. To choose a destination in
an app, pass the file URL from your save/export flow. Encoding and file system errors
are thrown with readable descriptions. For large images, run conversion and saving
off the main actor to keep your app responsive.

PNG export preserves transparency. These helpers work with generated images;
reference images prepared for editing use a different format. If you construct
pixel data yourself, see the [pixel requirements](../../ImageGeneration.md#options-and-editing).
Invalid pixel data throws `ModelImageEncodingError`. PNG files do not include
generation settings, so keep `result.seed` and other settings separately if needed.

## Options and editing

Use `ImageGenerationParameters` to set dimensions, a seed, sampling steps, guidance, or a negative prompt. Omit `steps` and `guidance` to use checkpoint defaults; output dimensions default to 1024 × 1024. Supported sizes and options vary by model.

For image editing, pass `referenceImages: [ModelImageData]` to `generate`. Check `model.supportsImageEditing` before submitting references. Reference pixels must be normalized to `[-1, 1]` in NCHW layout; generated pixels use HWC layout and values in `0...255`.

Keep the operation handle to cancel a request:

```swift
let operation = model.generate(prompt: "A misty forest at sunrise.")
operation.cancel()
```

See the [image generation guide](../../ImageGeneration.md) for reference inputs, cancellation, and the supported model families. The [FLUX.2 README](../Flux2/README.md) includes CLI commands, editing examples, and model-specific size constraints.
