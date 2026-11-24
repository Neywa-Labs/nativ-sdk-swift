# Mage-Flow

Mage-Flow generates images at native resolution. Mage-Flow-Edit takes an
instruction and one or more reference images. Refer to the inputs as
`Image 1`, `Image 2`, and so on in the order you attach them.

| Checkpoint | Task | Default steps | Guidance |
| --- | --- | ---: | ---: |
| [Mage-Flow](https://huggingface.co/mage-flow-community/Mage-Flow) | Generation | 20 | 5 |
| [Mage-Flow-Base](https://huggingface.co/mage-flow-community/Mage-Flow-Base) | Generation | 30 | 5 |
| [Mage-Flow-Turbo](https://huggingface.co/mage-flow-community/Mage-Flow-Turbo) | Generation | 4 | 1 |
| [Mage-Flow-Edit](https://huggingface.co/mage-flow-community/Mage-Flow-Edit) | Editing | 30 | 5 |
| [Mage-Flow-Edit-Base](https://huggingface.co/mage-flow-community/Mage-Flow-Edit-Base) | Editing | 30 | 5 |
| [Mage-Flow-Edit-Turbo](https://huggingface.co/mage-flow-community/Mage-Flow-Edit-Turbo) | Editing | 4 | 1 |

Output dimensions must be multiples of 16 between 512 and 2048 pixels per side.
The SDK accepts the original Diffusers checkpoints and MLX conversions with
`mlx_mage_flow.json`. For a local copy of an original checkpoint, keep its
checkpoint name as the directory name so the SDK can select the task and
sampling defaults. A renamed copy can declare its `variant` in
`mlx_mage_flow.json`, for example `mage-flow-edit-turbo`.

## CLI

```sh
swift run nativ generate_image --model mage-flow-community/Mage-Flow-Turbo \
  --prompt "A red ceramic teapot on an oak table in soft window light" \
  --size 768x512 --seed 42 --output teapot.png
```

```sh
swift run nativ generate_image --model mage-flow-community/Mage-Flow-Edit-Turbo \
  --reference-image scene.png --reference-image object.png \
  --prompt "Place the object from Image 2 on the table in Image 1" \
  --size 768x512 --seed 42 --output edit.png
```

Use `--steps`, `--guidance`, and `--negative-prompt` to override the sampling
defaults.

## Swift

```swift
import Foundation
import NativSDK

let model = try await NativImageModel.loadModel(
    from: .repository("mage-flow-community/Mage-Flow-Edit-Turbo"))
let result = try await model.generate(
    prompt: "Place the object from Image 2 on the table in Image 1",
    referenceImages: [scene, object],
    parameters: .init(seed: 42, width: 768, height: 512)
).result()
try result.image.savePNG(to: URL(fileURLWithPath: "edit.png"))
```

Pass the ordered references as normalized `ModelImageData`. See the
[image generation guide](../../ImageGeneration.md) for the input layout.
Use a Mage-Flow generation checkpoint to generate directly from text.

## License

The checkpoints are released under the [MIT license](https://huggingface.co/mage-flow-community/Mage-Flow-Turbo).
See [Microsoft's Mage-Flow project](https://github.com/microsoft/Mage-Flow) for
the reference implementation.
