# Image generation

Generate or edit one image with `NativImageModel`. Load a Hugging Face repository
with `.repository(id, revision:)` or a local checkpoint with `.directory(url)`.

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

Output dimensions default to 1024 × 1024. Omit `steps` and `guidance` to use the
model's defaults. Use a seed for repeatable requests with the same model and settings.

Use `result.image.cgImage()` to display the output, `pngData()` to get encoded bytes,
or `savePNG(to:)` to save a PNG with transparency preserved. See
[display and export](Models/ImageGeneration/README.md#display-and-export) for UI integration
and file-saving behavior.

## Supported models

| Family | Variants | Generation and editing |
| --- | --- | --- |
| [Flux2](Models/Flux2/README.md) | Klein 4B, 9B, Base 4B/9B, 9B KV | Image generation and multiple references; reference caching on the KV variant |
| [FLUX.3](Models/Flux3/README.md) | Image Klein, Image Klein Base | Image generation, up to ten ordered references, optional prompt expansion and bounding boxes |
| [Bonsai](Models/Bonsai/README.md) | Ternary | Image generation, negative prompts |
| [Ideogram4](Models/Ideogram4/README.md) | FP8 | RGB generation, structured JSON captions, optional caption expansion, configurable sampler and guidance schedule |
| [MiniMax-H3-Image](Models/MiniMaxH3Image/README.md) | BF16, 4-bit | Six-step RGB generation and editing with up to 16 ordered references, optional caption expansion |
| [Mage-Flow](Models/MageFlow/README.md) | Base, standard, Turbo | Native-resolution generation and multi-reference editing on the corresponding checkpoints |
| [ERNIE-Image](Models/ErnieImage/README.md) | Base, Turbo | Image generation, single-image editing with adjustable strength, optional prompt enhancement |
| [Z-Image](Models/ZImage/README.md) | Base, Turbo | Image generation and single-image editing with adjustable strength |
| [Qwen-Image](Models/QwenImage/README.md) | 2.1 | RGB generation and RGBA multiple-reference editing, reference caching and negative prompts |
| [MOSS-Image](Models/MOSSImage/README.md) | Base, SFT | RGB generation from captions and JSON layouts, negative prompts and skip-layer guidance |
| [Ming-Image](Models/MingImage/README.md) | 0.1 Design | RGBA generation for text-rich designs |
| [LLaDA-Image](Models/LLaDAImage/README.md) | Base, Turbo; BF16 or FP8 | RGB generation, single-reference editing, deterministic or stochastic sampling |


## Options and editing

`ImageGenerationParameters` controls dimensions, steps, guidance, seed, and a
negative prompt. Model-specific options and valid sizes are described in the
family guides above.

For editing, pass `referenceImages: [ModelImageData]` to `generate`. Check
`model.supportsImageEditing` first. Some models accept multiple references; each
request still returns one image.
Bonsai, Ming, and Ideogram4 do not accept reference images.

Read `result.image.tensor` for a shareable `MaterializedArray` in the pipeline's
native dtype, or `result.image.pixels` for Swift `[Float]`. Output pixels are HWC
RGB or RGBA in `0...255`. To edit an image already in MLX, wrap the reference with
`try ModelImageData(tensor: normalizedImage)`. Reference pixels must use
NCHW `[1, 3 or 4, height, width]` layout and values in `[-1, 1]`. Qwen editing
preserves reference aspect ratio and alpha when resizing; `referenceResolution`
controls the amount of detail retained from reference images.

Keep the operation handle to cancel generation:

```swift
let operation = model.generate(prompt: "A misty forest at sunrise.")
// When the user stops the request:
operation.cancel()
```

## Prompt enhancement

Prompt expansion is off by default. Set `parameters.promptExpansion` to
`.defaultModel` for the ERNIE-Image, FLUX.3, or MiniMax-H3-Image caption model or to
`.model(.repository("owner/model"))` or `.model(.directory(url))` to supply a
compatible model. Ideogram4 uses a model you supply and sends structured JSON
captions directly to the image pipeline. With `autoJSONCaption` enabled,
Ideogram4 falls back to a basic caption if expansion fails. The expanded caption
or fallback information is available in `result.metadata`.

The CLI accepts `--upsample-prompt` for a pipeline's default expander and
`--prompt-expansion-model <repo|path>` for a specific model.

## CLI

```sh
swift run nativ image --model Tongyi-MAI/Z-Image-Turbo \
  --prompt "An ink drawing of a small observatory" --output observatory.png
swift run nativ image --model black-forest-labs/FLUX.2-klein-4B \
  --task edit --image source.png --prompt "Replace the background with a forest" \
  --output edit.png
```

Run `swift run nativ image --help` for options. The family guides cover controls
such as `--strength`, `--negative-prompt`, `--reference-resolution`, and prompt enhancement.

## Checkpoint metadata

Keep the complete checkpoint when copying models: weights, `model_index.json`,
component configurations, tokenizer files, and conversion metadata. Missing files
can prevent loading or disable optional features.
