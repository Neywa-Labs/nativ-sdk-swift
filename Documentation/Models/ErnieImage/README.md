# ERNIE-Image

ERNIE-Image generates RGB images from text and edits a single reference image.
The same checkpoint supports both tasks. Load the official Diffusers snapshots
directly or use a local checkpoint converted with `mlx-vlm`.

| Checkpoint | Default schedule steps | Generation guidance | Editing guidance |
| --- | ---: | ---: | ---: |
| [ERNIE-Image Base](https://huggingface.co/baidu/ERNIE-Image) | 50 | 4 | 4 |
| [ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo) | 8 | 1 | 3 |

Width and height must be positive multiples of 16. Editing resizes the reference
to the output dimensions and adds noise to its latent representation.
`strength` controls how much of the denoising schedule runs, from a small edit
near zero to the full schedule at 1. The default is 0.6. The result reports the
number of denoising steps performed. Negative prompts take effect when guidance
is greater than 1.

## CLI

```sh
swift run nativ image --model baidu/ERNIE-Image-Turbo \
  --prompt "An ink drawing of a cat on a tiled roof at dusk" \
  --size 1024x768 --upsample-prompt --seed 42 --output cat.png

swift run nativ image --model baidu/ERNIE-Image-Turbo \
  --reference-image cat.png --prompt "A tiled roof at dawn, tinted rose and gold" \
  --size 1024x768 --strength 0.6 --seed 42 --output cat-at-dawn.png
```

## Swift

```swift
import Foundation
import NativSDK

let model = try await NativImageModel.loadModel(from: .repository("baidu/ERNIE-Image-Turbo"))
let parameters = ImageGenerationParameters(
    seed: 42, width: 1024, height: 768, promptExpansion: .defaultModel)
let result = try await model.generate(
    prompt: "An ink drawing of a cat on a tiled roof at dusk",
    parameters: parameters
).result()
try result.image.savePNG(to: URL.documentsDirectory.appending(path: "cat.png"))
```

To edit, supply one normalized `ModelImageData` in `referenceImages` and set
`parameters.strength`. Keep the VAE encoder in local or converted checkpoints
to enable editing. Check `model.supportsImageEditing` for the loaded checkpoint.

## Prompt enhancement

Set `promptExpansion` to `.defaultModel` or use `--upsample-prompt` to enable the
bundled enhancer. For the official repositories, the SDK downloads it on first
use. With a local checkpoint, keep the bundled `pe/` and `pe_tokenizer/` directories.
You can supply compatible standalone ERNIE-Image PE assets with
`.model(.directory(url))`, `.model(.repository("owner/model"))`, or the CLI's
`--prompt-expansion-model <repo|path>` flag.

The enhancer receives the prompt and output dimensions. For editing, the image
itself goes to the image pipeline. The expanded caption is returned as
`result.metadata["revised_prompt"]`.
