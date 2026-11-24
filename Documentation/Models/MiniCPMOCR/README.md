# MiniCPM-OCR 1.0

MiniCPM-OCR 1.0 is a 1.3B model that converts document images into Markdown,
including text, tables, and formulas.

## Model

- [openbmb/MiniCPM-OCR-1.0](https://huggingface.co/openbmb/MiniCPM-OCR-1.0)

The original BF16 checkpoint loads directly through the native Swift model and
processor. Its weights occupy approximately 2.6 GB; inference needs additional
memory for image features, activations, and caches.

## CLI

```sh
swift run nativ generate \
  --model openbmb/MiniCPM-OCR-1.0 \
  --image ./page.png \
  --prompt "Convert this page to markdown." \
  --temperature 0 --max-tokens 4096
```

## Swift

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(
    from: .repository("openbmb/MiniCPM-OCR-1.0"))
let imageURL = URL(fileURLWithPath: "./page.png")
let input = UserInput(chat: [
    .user("Convert this page to markdown.", images: [.url(imageURL)])
])
let text = try await model.generate(
    input: input,
    options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 4096))
).response()
print(text)
```

## Compression and image sizing

The checkpoint defaults to `4x` visual compression. Use `16x` to reduce the
visual token count, with a possible loss of small-text detail. Pass processor
options in `UserInput.additionalContext`:

```swift
let input = UserInput(
    chat: [.user("Convert this page to markdown.", images: [.url(imageURL)])],
    additionalContext: [
        "downsample_mode": "16x",
        "adaptive_small_image_upsize": true,
        "adaptive_tile_upsize": true,
        "disable_small_image_upsize": true,
    ])
```

These settings apply only to that request. The processor also reads defaults
from the checkpoint's image configuration.

| Option | Default | Behavior |
| --- | --- | --- |
| `downsample_mode` | `"4x"` | Select `"4x"` or `"16x"` visual compression. |
| `max_slice_nums` | `9` | Bound the local crop grid; `0` disables slicing. |
| `slice_mode` | `true` | Enable local crops for larger images. |
| `use_image_id` | `true` | Include image identifiers in the prompt. |
| `disable_small_image_upsize` | `false` | Keep small images at native size unless adaptive rules request upsize. |
| `adaptive_small_image_upsize` | `false` | Detect small dense text and distinguish tiny crops from other images. |
| `adaptive_tile_upsize` | `false` | Apply adaptive sizing to local crops when adaptive image sizing is enabled. |

Adaptive sizing keeps small dense text at the full resolution budget, assigns
tiny crops a smaller budget, and can retain native resolution for other crops.
The advanced thresholds are `adaptive_upsize_font_px` (12),
`adaptive_upsize_dense_per_kpx` (2), `adaptive_upsize_tiny_pixels` (6272), and
`adaptive_upsize_tiny_target_resolution` (84). All are per-request options too.

## Notes

The processor preserves chat history and the checkpoint's chat template, with
thinking disabled by default. Pass `"enable_thinking": true` in
`additionalContext` to request a thinking turn.

The model accepts one input sequence per request and supports chunked prefill.

The model and reference processor use Apache-2.0; see the
[upstream model](https://huggingface.co/openbmb/MiniCPM-OCR-1.0) and
[MiniCPM-OCR project](https://github.com/OpenBMB/MiniCPM-OCR).
The implementation follows the
[Python MLX port](https://github.com/Neywa-Labs/mlx-vlm/tree/e9b3c21e/mlx_vlm/models/minicpm_ocr).
