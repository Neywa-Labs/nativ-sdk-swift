# Granite Vision 3.2

Granite Vision 3.2 is an IBM vision-language model for image understanding.

- Example checkpoint: `mlx-community/granite-vision-3.2-2b-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/granite-vision-3.2-2b-4bit \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 128
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/granite-vision-3.2-2b-4bit")
let input = UserInput(chat: [
    .user("Describe this image.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

Use image prompts for description and visual question answering.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
