# Aya Vision

Aya Vision is a Cohere vision-language family for image understanding and
multilingual visual question answering.

- Example checkpoint: `mlx-community/aya-vision-8b-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/aya-vision-8b-4bit \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 128
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/aya-vision-8b-4bit")
let imageURL = URL(fileURLWithPath: "image.jpg")
let input = UserInput(chat: [
    .user("Describe this image.", images: [.url(imageURL)])
])

try await generateText(source: source, input: input)
```

## Notes

Use image prompts for multilingual visual question answering. Choose a compatible MLX checkpoint.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
