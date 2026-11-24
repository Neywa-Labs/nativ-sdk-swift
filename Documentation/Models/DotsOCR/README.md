# dots.ocr

dots.ocr reads text and understands document images.

- Example checkpoint: `mlx-community/dots.ocr-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/dots.ocr-4bit \
  --image ./document.png \
  --prompt "Extract all text from this image." \
  --max-tokens 2000
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/dots.ocr-4bit")
let input = UserInput(chat: [
    .user("OCR:", images: [.url(URL(fileURLWithPath: "document.png"))])
])

try await generateText(source: source, input: input)
```

## Notes

Use OCR and document-understanding prompts with image attachments.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
