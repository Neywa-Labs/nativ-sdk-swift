# Florence 2

Florence 2 supports captioning, OCR, object detection, and structured vision tasks.

- Example checkpoint: `mlx-community/Florence-2-base-ft-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/Florence-2-base-ft-4bit \
  --image ./image.jpg \
  --prompt "<CAPTION>" \
  --max-tokens 128
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/Florence-2-base-ft-4bit")
let input = UserInput(chat: [
    .user("<CAPTION>", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Prompt Examples

- `"<CAPTION>"`
- `"<DETAILED_CAPTION>"`
- `"<OCR>"`
- `"<OD>"`

## Notes

Use the task tokens above to select captioning, OCR, or object-detection output.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
