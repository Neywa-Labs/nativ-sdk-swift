# Kimi VL

Kimi VL is a vision-language model for image understanding and reasoning.

- Example checkpoint: `mlx-community/Kimi-VL-A3B-Thinking-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/Kimi-VL-A3B-Thinking-4bit \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/Kimi-VL-A3B-Thinking-4bit")
let input = UserInput(chat: [
    .user("Describe this image.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

Use image prompts for visual reasoning.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
