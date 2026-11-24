# Mistral Large 3

Mistral Large 3 supports text conversations and image understanding.

## Model repository

- [mlx-community/Mistral-Large-3-675B-Instruct-2512-mlx-4bit](https://huggingface.co/mlx-community/Mistral-Large-3-675B-Instruct-2512-mlx-4bit): 4-bit MLX checkpoint with text and vision weights.

Load this repository directly in Nativ SDK. It includes the configuration,
weights, tokenizer, and chat template; no local conversion is required.

## CLI

```sh
swift run nativ generate \
  --model mlx-community/Mistral-Large-3-675B-Instruct-2512-mlx-4bit \
  --image ./image.jpg --prompt 'Describe this image.' \
  --max-tokens 128
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(
    from: .repository("mlx-community/Mistral-Large-3-675B-Instruct-2512-mlx-4bit"))
let request = model.generate(
    input: UserInput(chat: [
        .user("Describe this image.", images: [.url(URL(fileURLWithPath: "./image.jpg"))])
    ]),
    options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 128)))
for try await event in request {
    if case let .chunk(text) = event { print(text, terminator: "") }
}
```

## Checkpoint requirements

For a local copy, use `.directory(url)` and keep `config.json`, all safetensors,
`tokenizer.json`, `tokenizer_config.json`, and the chat template together.
The original `mistralai/Mistral-Large-3-675B-Instruct-2512` distribution uses
`params.json` and `tekken.json` and still requires conversion.

Image requests support one image at a time. Allow sufficient unified memory for
this large model and its conversation history.
