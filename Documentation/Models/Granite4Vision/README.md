# Granite 4 Vision

Granite 4 Vision is the newer IBM Granite vision-language family.

- Example checkpoint: `mlx-community/granite-4.0-3b-vision-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/granite-4.0-3b-vision-4bit \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 128
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/granite-4.0-3b-vision-4bit")
let input = UserInput(chat: [
    .user("Describe this image.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

Use image prompts for description and visual question answering.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
