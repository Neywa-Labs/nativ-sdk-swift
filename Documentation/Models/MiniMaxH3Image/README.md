# MiniMax-H3-Image

MiniMax-H3-Image generates images from a text caption and edits images using up
to 16 ordered references. Label them `Picture 1`, `Picture 2`, and so on in the
caption to refer to specific inputs. The released model uses six sampling steps
and guidance 1.

The [4-bit MLX checkpoint](https://huggingface.co/nativ-community/MiniMax-H3-Image-4bit)
uses about 54.5 GB of disk space. The SDK also loads the original Diffusers
components and their shared Qwen3-VL conditioner. [MiniMax checkpoint](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)

## CLI

```sh
swift run nativ generate_image --model nativ-community/MiniMax-H3-Image-4bit \
  --prompt "A blue ceramic teacup on a pale wooden table in soft side light." \
  --width 1024 --height 1024 --seed 11 --output teacup.png
```

```sh
swift run nativ generate_image --model nativ-community/MiniMax-H3-Image-4bit \
  --reference-image mug.jpg --reference-image table.jpg \
  --prompt "Place the mug from Picture 1 on the table from Picture 2." \
  --output edit.png
```

The default internal canvas area is 4,194,304 pixels. The pipeline resizes the
result to the requested dimensions. Use `--h3-canvas-area` to set the minimum
internal area explicitly.

Pass `--upsample-prompt` to expand a short instruction before generation or
editing. This loads a separate caption model and keeps the requested output
dimensions. Use `--prompt-expansion-model` to enable expansion with a
compatible local checkpoint or repository.
[MiniMax-H3-Image-PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE)

## Swift

```swift
import Foundation
import NativSDK

let model = try await NativImageModel.loadModel(
    from: .repository("nativ-community/MiniMax-H3-Image-4bit"))
let result = try await model.generate(
    prompt: "A blue ceramic teacup on a pale wooden table in soft side light.",
    parameters: .init(seed: 11, width: 1024, height: 1024)
).result()
try result.image.savePNG(to: URL(fileURLWithPath: "teacup.png"))
```

For editing, pass the ordered inputs as `referenceImages`. Set
`parameters.promptExpansion = .defaultModel` to enable caption expansion and
read the expanded caption and suggested aspect ratio from `result.metadata`.
Use `.model(.directory(url))` or `.model(.repository("owner/model"))` to
supply a compatible expander.

## License

The model weights and caption templates are distributed under the MiniMax
community license. [License](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE)
