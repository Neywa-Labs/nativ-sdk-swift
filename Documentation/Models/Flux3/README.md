# FLUX.3 Image Klein

FLUX.3 Image Klein generates images from text and edits them using up to ten
ordered reference images. Use “the first image”, “the second image”, and so on
to refer to particular inputs.

| Native BF16 checkpoint | Default steps | Guidance |
| --- | ---: | ---: |
| [Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) | 16 | 1 |
| [Klein Base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base) | 50 | 3.5 |

Klein uses a distilled flow map with guidance 1. Klein Base uses classifier-free
guidance and accepts a negative prompt. Each output side must be a multiple of
16 between 64 and 4096 pixels.

Accept the checkpoint access conditions on Hugging Face and authenticate before
the first download. A local native checkout can also be supplied directly. Keep
`config.json`, `model.safetensors`, `ae.safetensors`, and the full `text_encoder/`
folder together.

## CLI

```sh
swift run nativ generate_image --model black-forest-labs/flux-3-image-klein \
  --prompt "A cobalt blue ceramic teapot on a pale oak table" \
  --upsample-prompt --size 512x512 --seed 42 --output teapot.png
```

```sh
swift run nativ generate_image --model black-forest-labs/flux-3-image-klein \
  --reference-image teapot.png --reference-image flowers.jpg \
  --prompt "Put the flowers from the second image into the teapot from the first" \
  --upsample-prompt --size 512x512 --output edit.png
```

`--upsample-prompt` loads the [Black Forest Labs caption model](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup).
Use `--prompt-expansion-model <repo|path>` to supply a compatible alternative.
Omit both flags to generate from a finished caption. The native text encoder
can accept up to 4096 tokens. Use `--max-length` for captions longer than the
SDK's default 512-token budget.

For layout control, append a JSON table using coordinates
`[top, left, bottom, right]` on a 0–1000 canvas:

```text
A teapot on the left of an oak table [{"id":"pot","bbox":[200,50,800,450],"desc":"red ceramic teapot"}]
```

With expansion enabled, the caption model adds prose and preserves the supplied
boxes. In a finished caption, cite each box with its ID in angle brackets,
such as `<pot>`.

## Swift

```swift
import Foundation
import NativSDK

let model = try await NativImageModel.loadModel(
    from: .repository("black-forest-labs/flux-3-image-klein"))
let result = try await model.generate(
    prompt: "A cobalt blue ceramic teapot on a pale oak table",
    parameters: .init(seed: 42, width: 512, height: 512, promptExpansion: .defaultModel)
).result()
try result.image.savePNG(to: URL(fileURLWithPath: "teapot.png"))
```

Pass ordered `referenceImages` to edit. Set `promptExpansion` to `.model(source)`
to supply your own expander. Read the expanded caption at
`result.metadata["revised_prompt"]`.

## License

The image weights use the [FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md).
Black Forest Labs' [reference code](https://github.com/black-forest-labs/flux-3-image-klein)
and the prompt upsampler are Apache-2.0.
