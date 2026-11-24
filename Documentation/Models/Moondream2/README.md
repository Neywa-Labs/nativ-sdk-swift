# Moondream2

Moondream2 is a compact vision-language model for captioning and VQA.

- Example checkpoint: `vikhyatk/moondream2`
- Tokenizer source: `moondream/starmie-v1`

## CLI

```sh
swift run nativ generate \
  --model vikhyatk/moondream2 \
  --image ./image.jpg \
  --prompt "Describe this image in detail." \
  --max-tokens 160
```

## Swift

```swift
let source = ModelSource.repository("vikhyatk/moondream2")
let input = UserInput(chat: [
    .user("Describe this image in detail.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

- Region tasks are not exposed through text generation yet.
- Multi-crop image inputs are supported automatically.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
