# MiniCPM-OCR Jev

Jev scores whether candidate OCR text and its rendered image match an original
document crop.

## Model

- [openbmb/MiniCPM-OCR-Jev](https://huggingface.co/openbmb/MiniCPM-OCR-Jev)

The original BF16 checkpoint loads directly and occupies approximately 4.43 GB.
Inference requires additional working memory. Native and Python MLX checkpoints,
including quantized backbone weights, are also supported.

## Swift

Supply the original crop, candidate text, and an image rendered from that text.
Render the candidate separately with a renderer suitable for its content (text,
formulas, or tables).

```swift
import Foundation
import NativSDK

let scorer = try await NativOCRScorer.loadModel(
    from: .repository("openbmb/MiniCPM-OCR-Jev"))
let result = try await scorer.score(
    original: URL(fileURLWithPath: "./original-crop.png"),
    text: "Invoice number: 1042\nTotal: $27.50",
    rendering: URL(fileURLWithPath: "./rendered-candidate.png")
).result()

print(result.consistentProbability)
print(result.inconsistentLogit, result.consistentLogit)
print(result.tokenCount)
```

The image arguments also accept `UserInput.Image`, including `.ciImage` for decoded
images. Requests execute exclusively on the loaded scorer. Retain the returned
`ModelOperation` to cancel a request or await its result.

## Input and scores

Pass the candidate text with its original whitespace. The scorer prepares the
prompt and images automatically. Use the request's `processing` argument to override
the checkpoint's pixel budgets. Large crops can require substantial vision memory.

Class 1 means consistent; class 0 means inconsistent. The result contains both
raw logits, their softmax probabilities, and the input token count. Choose a
decision threshold using representative documents and your deployment's precision
and rendering pipeline.

The default conversion policy keeps the scoring head at checkpoint precision.

## License and references

The model and reference scoring code use Apache-2.0; see the
[model card](https://huggingface.co/openbmb/MiniCPM-OCR-Jev),
[OpenBMB project](https://github.com/OpenBMB/MiniCPM-OCR), and
[Python MLX port](https://github.com/Neywa-Labs/mlx-vlm/tree/e9b3c21e/mlx_vlm/models/minicpm_ocr_jev).
