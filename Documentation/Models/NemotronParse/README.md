# Nemotron-Parse

Nemotron-Parse reads document images and can return text, layout markers, and bounding-box annotations.

## Model repositories

| Repository | Notes |
| --- | --- |
| [mlx-community/Nemotron-Parse-2.0-8bit](https://huggingface.co/mlx-community/Nemotron-Parse-2.0-8bit) | 2.0, 8-bit checkpoint |
| [mlx-community/Nemotron-Parse-2.0-4bit](https://huggingface.co/mlx-community/Nemotron-Parse-2.0-4bit) | 2.0, 4-bit checkpoint |
| [mlx-community/Nemotron-Parse-v1.2-8bit](https://huggingface.co/mlx-community/Nemotron-Parse-v1.2-8bit) | v1.2, 8-bit checkpoint |
| [mlx-community/Nemotron-Parse-v1.2-4bit](https://huggingface.co/mlx-community/Nemotron-Parse-v1.2-4bit) | v1.2, 4-bit checkpoint |

Task prompts such as `<predict_bbox>` and `<output_markdown>` currently have no
effect. Use the examples below for document image-to-text output.

## CLI

```sh
swift run nativ generate \
  --model mlx-community/Nemotron-Parse-2.0-8bit \
  --image ./image.jpg \
  --prompt "Parse this document." \
  --max-tokens 1024
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(from: .repository("mlx-community/Nemotron-Parse-2.0-8bit"))
let input = UserInput(prompt: .text("Parse this document."), images: [.url(URL(fileURLWithPath: "./image.jpg"))])
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

Pass exactly one document image. PDF files, video, and audio are unsupported; render each PDF page to an image first.
