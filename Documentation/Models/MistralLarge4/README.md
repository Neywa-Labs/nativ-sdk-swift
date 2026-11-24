# Mistral Large 4

Mistral Large 4 is a 1T-parameter mixture-of-experts model for text conversations,
tool calling, and image understanding. Image requests accept one image at a time.

## Checkpoints

- [Mistral Large 4 FP8](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B)
- [Mistral Large 4 NVFP4](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B-NVFP4)
- [Mistral Large 4 Text 4-bit](https://huggingface.co/mlx-community/Mistral-Large-4-1T-A52B-Text-4bit)

Keep the checkpoint configuration (`config.json` or `params.json`), all weight
shards, tokenizer, chat template, and processor files together. The text-only
checkpoint can be loaded with the same API and accepts text prompts.

Allow enough unified memory for the resident weights and conversation cache.
Loading expands block-FP8 matrices to BF16 and keeps NVFP4 experts packed.
Generation uses a growing, full-precision cache on a single machine.

## CLI

```sh
swift run nativ generate \
  --model /path/to/mistral-large-4 \
  --image ./image.jpg --prompt 'Describe this image.' \
  --max-tokens 128
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(
    from: .directory(URL(fileURLWithPath: "/path/to/mistral-large-4")))
let request = model.generate(
    input: UserInput(chat: [
        .user("Describe this image.", images: [.url(URL(fileURLWithPath: "./image.jpg"))])
    ]),
    options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 128)))
for try await event in request {
    if case let .chunk(text) = event { print(text, terminator: "") }
}
```
