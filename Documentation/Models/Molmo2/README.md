# Molmo2

Molmo2 supports image description, visual question answering, and reasoning.

- Example checkpoint: `mlx-community/Molmo2-8B-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/Molmo2-8B-4bit \
  --image ./image.jpg \
  --prompt "Describe this image in detail." \
  --max-tokens 160
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/Molmo2-8B-4bit")
let input = UserInput(chat: [
    .user("Describe this image in detail.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

- Use image prompts for description, VQA, and visual reasoning.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
