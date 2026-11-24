# DeepSeek OCR

DeepSeek OCR reads documents, converts pages to Markdown, and locates text in images.

- Example checkpoint: `mlx-community/DeepSeek-OCR-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/DeepSeek-OCR-4bit \
  --image ./document.png \
  --prompt "<|grounding|>Convert the document to markdown." \
  --max-tokens 2000
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/DeepSeek-OCR-4bit")
let input = UserInput(chat: [
    .user("<|grounding|>OCR this image.", images: [.url(URL(fileURLWithPath: "document.png"))])
])

try await generateText(source: source, input: input)
```

## Prompt Examples

- `"<|grounding|>Convert the document to markdown."`
- `"<|grounding|>OCR this image."`
- `"Free OCR."`
- `"Locate <|ref|>Total assets<|/ref|> in the image."`

## Notes

- Bounding boxes use normalized 0-1000 image coordinates.
- Prompts can ask for full-page OCR, markdown conversion, or locating specific
  referenced text.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
