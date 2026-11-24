# Nemotron 3.5 Parse

Nemotron 3.5 Parse reads document pages into text, coordinates, semantic classes,
and document markup.

## Model

- [nvidia/Nemotron-3.5-Parse](https://huggingface.co/nvidia/Nemotron-3.5-Parse)

The original FP32 checkpoint loads directly and occupies approximately 2.52 GB.
Converted MLX checkpoints and quantized decoder weights are also supported. The
default quantization policy preserves the vision tower at checkpoint precision.

## CLI

```sh
swift run nativ generate \
  --model nvidia/Nemotron-3.5-Parse \
  --image ./page.png \
  --prompt '</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>' \
  --temperature 0 --max-tokens 8744 \
  --repetition-penalty 1.1 --repetition-context-size 0
```

## Swift

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(
    from: .repository("nvidia/Nemotron-3.5-Parse"))
let prompt = "</s><s><predict_bbox><predict_classes>"
    + "<output_markdown><predict_text_in_pic>"
let input = UserInput(
    prompt: .text(prompt),
    images: [.url(URL(fileURLWithPath: "./page.png"))])
let rawOutput = try await model.generate(
    input: input,
    options: .init(
        prefill: .init(stepSize: 128),
        sampling: .init(temperature: 0, repetition: .init(1.1, contextSize: 0)),
        output: .init(maxTokens: 8744))
).response()
print(rawOutput)
```

Keep coordinate and class markers in the returned text for downstream document
assembly. NVIDIA provides postprocessing and multi-pass orchestration alongside
the checkpoint.

## Input and runtime

Supply one RGB page and a task prompt as raw text or a single structured user
message. Leave the prompt empty to use the default shown above.

Pages shrink to fit the checkpoint's `final_size` (2048×1664 by default) and
retain their aspect ratio. Smaller pages retain their original resolution.

The 16,384-position context limit covers the image, prompt, and generated
tokens. The model supports chunked prefill and full-context caching.
MTP and DFlash2 drafting are unsupported.

## License and references

The model materials and reference forward implementation use
[OpenMDW-1.1](https://huggingface.co/nvidia/Nemotron-3.5-Parse/blob/main/LICENSE).
See NVIDIA's [model card](https://huggingface.co/nvidia/Nemotron-3.5-Parse),
[tokenizer attribution](https://huggingface.co/nvidia/Nemotron-3.5-Parse/blob/main/TOKENIZER_ATTRIBUTION.md),
[third-party notices](https://huggingface.co/nvidia/Nemotron-3.5-Parse/blob/main/THIRD_PARTY_NOTICES.md),
and the [Python MLX port](https://github.com/Neywa-Labs/mlx-vlm/tree/main/mlx_vlm/models/nemotron_parse_v3_5).
