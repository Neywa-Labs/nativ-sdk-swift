# IndicOCR

IndicOCR extracts text from document images. Nativ SDK supports the OCR stage of the model.

## Model repositories

- [HashNuke/indic-ocr-mlx](https://huggingface.co/HashNuke/indic-ocr-mlx): combined
  repository; select the `weights/ocr` stage.
- [bodhan-ai/indic-ocr](https://huggingface.co/bodhan-ai/indic-ocr): original release.

## CLI

Download the OCR stage, then point the CLI at that concrete subdirectory.
`ModelSource.repository` selects a repository root, not a nested stage:

```sh
hf download HashNuke/indic-ocr-mlx \
  --include "weights/ocr/*" --local-dir ./models/indic-ocr

swift run nativ generate \
  --model ./models/indic-ocr/weights/ocr \
  --image ./document.png --prompt 'Transcribe the text in this image.' \
  --max-tokens 1024
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(
    from: .directory(URL(fileURLWithPath: "./models/indic-ocr/weights/ocr")))
let request = model.generate(
    input: UserInput(chat: [
        .user("Transcribe the text in this image.", images: [.url(URL(fileURLWithPath: "./document.png"))])
    ]),
    options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 1024)))
for try await event in request {
    if case let .chunk(text) = event { print(text, terminator: "") }
}
```

## Checkpoint requirements

The directory must contain the OCR stage's `text_config`, `vision_config`,
weights, tokenizer, and chat template. Combined layout/OCR checkpoints and
wrapper repositories containing `stages` are rejected: document layout detection
and the two-stage document parsing pipeline are not included.
