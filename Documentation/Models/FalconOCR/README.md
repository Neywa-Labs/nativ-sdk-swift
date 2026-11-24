# Falcon OCR

Falcon OCR is an OCR model for plain text extraction and document-oriented
image understanding.

- Example checkpoint: `mlx-community/Falcon-OCR-bf16`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/Falcon-OCR-bf16 \
  --image ./document.png \
  --prompt "plain" \
  --max-tokens 2000
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/Falcon-OCR-bf16")
let input = UserInput(chat: [
    .user("plain", images: [.url(URL(fileURLWithPath: "document.png"))])
])

try await generateText(source: source, input: input)
```

## Notes

- Use `plain` for basic text extraction from document images.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
