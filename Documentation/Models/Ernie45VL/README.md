# ERNIE 4.5 VL

ERNIE 4.5 VL supports image understanding and visual reasoning.

- Example checkpoint: `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit")
let input = UserInput(chat: [
    .user("Describe this image.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

- Use image prompts for ERNIE 4.5 VL visual reasoning.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
