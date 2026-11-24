# PaddleOCR-VL

PaddleOCR-VL is a document OCR and vision-language model for multilingual
document understanding.

- Example checkpoint: `mlx-community/PaddleOCR-VL-1.5-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/PaddleOCR-VL-1.5-4bit \
  --image ./document.png \
  --prompt "Describe this image." \
  --max-tokens 2000
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/PaddleOCR-VL-1.5-4bit")
let input = UserInput(chat: [
    .user("Describe this image.", images: [.url(URL(fileURLWithPath: "document.png"))])
])

try await generateText(source: source, input: input)
```

## Notes

- Use document images and OCR-oriented prompts for multilingual page
  understanding.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
