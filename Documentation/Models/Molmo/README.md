# Molmo

Molmo is a vision-language family for image description, VQA, and visual
reasoning.

- Example checkpoint: `mlx-community/Molmo-7B-D-0924-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/Molmo-7B-D-0924-4bit \
  --image ./image.jpg \
  --prompt "Describe this image in detail." \
  --max-tokens 160
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/Molmo-7B-D-0924-4bit")
let input = UserInput(chat: [
    .user("Describe this image in detail.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

- Use image prompts for visual question answering, description, and reasoning.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
