# GLM-OCR

GLM-OCR reads text, equations, and tables from document images.

## Model repositories

| Repository | Notes |
| --- | --- |
| [mlx-community/GLM-OCR-bf16](https://huggingface.co/mlx-community/GLM-OCR-bf16) | BF16 checkpoint |
| [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | Original checkpoint |

## CLI

```sh
swift run nativ generate \
  --model mlx-community/GLM-OCR-bf16 \
  --image ./image.jpg \
  --prompt "Text Recognition:" \
  --max-tokens 1024
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(from: .repository("mlx-community/GLM-OCR-bf16"))
let input = UserInput(chat: [.user("Text Recognition:", images: [.url(URL(fileURLWithPath: "./image.jpg"))])])
let request = model.generate(
    input: input,
    options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 1024)))
for try await event in request {
    if case let .chunk(text) = event {
        print(text, terminator: "")
    }
}
```

## Notes

Use `Text Recognition:`, `Formula Recognition:`, or `Table Recognition:` for the corresponding task. For structured extraction, include the desired JSON fields in the prompt and validate the generated result in your application. Render PDF pages to images before inference.
