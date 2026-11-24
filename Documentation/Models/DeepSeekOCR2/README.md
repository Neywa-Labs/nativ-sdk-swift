# DeepSeek OCR 2

DeepSeek OCR 2 reads documents and extracts text from images.

- Example checkpoint: `mlx-community/DeepSeek-OCR-2-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/DeepSeek-OCR-2-4bit \
  --image ./document.png \
  --prompt "<|grounding|>OCR this image." \
  --max-tokens 2000
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/DeepSeek-OCR-2-4bit")
let input = UserInput(chat: [
    .user("<|grounding|>Convert the document to markdown.", images: [.url(URL(fileURLWithPath: "document.png"))])
])

try await generateText(source: source, input: input)
```

## Notes

- Use OCR prompts from the DeepSeek OCR README unless the checkpoint model card
  recommends a task-specific prompt.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
