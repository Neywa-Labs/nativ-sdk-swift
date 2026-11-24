# Phi-3.5 Vision

Phi-3.5 Vision is Microsoft's Phi vision-language model for image understanding.

- Example checkpoint: `mlx-community/Phi-3.5-vision-instruct-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/Phi-3.5-vision-instruct-4bit \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 160
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/Phi-3.5-vision-instruct-4bit")
let input = UserInput(chat: [
    .user("Describe this image.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

- HD image inputs are supported.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
