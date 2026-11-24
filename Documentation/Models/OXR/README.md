# OXR-1.0

OXR is a 1.3B model for recognizing text, formulas, HTML tables, and code in
document crops and whole pages.

## Model

- [SoMarkAI/OXR-1.0](https://huggingface.co/SoMarkAI/OXR-1.0)

The original BF16 checkpoint loads directly. Its weights occupy approximately
3.07 GB; image features, activations, and the generation cache require additional
memory. Native and Python MLX checkpoint namespaces are also supported.

## CLI

```sh
swift run nativ generate \
  --model SoMarkAI/OXR-1.0 \
  --image ./paragraph.png \
  --prompt "Recognize the content of the image." \
  --temperature 0 --max-tokens 4096
```

## Swift

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(
    from: .repository("SoMarkAI/OXR-1.0"))
let input = UserInput(chat: [
    .user("Recognize the content of the image.",
          images: [.url(URL(fileURLWithPath: "./paragraph.png"))])
])
let text = try await model.generate(
    input: input,
    options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 4096))
).response()
print(text)
```

## Task prompts

| Content | Prompt |
| --- | --- |
| Text or handwriting | `Recognize the content of the image.` |
| Formula | `Recognize the formula of the image into KaTex format.` |
| Table | `Recognize the table of the image into html format.` |
| Code | `Recognize the code of the image into markdown code format.` |
| Chemical structure | `Recognize the chemical structure of the image into SMILES.` |
| Whole page | `Recognize the whole page content of the image.` |

For a full document pipeline, supply crops from a layout model and assemble the
recognized regions. Layout detection and document orchestration are separate
components of the upstream system.

## Image processing

Images are resized within a default pixel budget of `448 * 448` to `960 * 960`.
Transparent areas are composited onto white.

Override pixel budgets for an individual request:

```swift
let imageURL = URL(fileURLWithPath: "./paragraph.png")
let input = UserInput(
    chat: [.user("Recognize the content of the image.", images: [.url(imageURL)])],
    processing: .init(minPixels: 448 * 448, maxPixels: 640 * 640))
```

Structured chats insert image placeholders automatically. Put `<image>` in a
message to control interleaving explicitly; placeholders must match the supplied
images. Chat history and the checkpoint template are preserved. Each request
accepts one input sequence with one or more images; chunked prefill is supported.

The default quantization policy keeps the vision encoder, visual-vocabulary head,
and visual embedding table dense at their checkpoint precision, while allowing
language weights to be quantized.

## License and references

The model uses the [OXR Open Source License](https://github.com/SoMarkAI/OXR/blob/main/LICENSE),
based on Apache-2.0 with additional terms. See the
[upstream project](https://github.com/SoMarkAI/OXR),
[model card](https://huggingface.co/SoMarkAI/OXR-1.0), and
[Python MLX implementation](https://github.com/Neywa-Labs/mlx-vlm/tree/e9b3c21e/mlx_vlm/models/oxr).
