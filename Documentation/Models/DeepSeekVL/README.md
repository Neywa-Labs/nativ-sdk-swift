# DeepSeek VL

DeepSeek VL supports image description and visual question answering.

- Example checkpoint: `mlx-community/deepseek-vl-1.3b-chat-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/deepseek-vl-1.3b-chat-4bit \
  --image ./image.jpg \
  --prompt "Describe this image in detail." \
  --max-tokens 160
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/deepseek-vl-1.3b-chat-4bit")
let input = UserInput(chat: [
    .user("Describe this image in detail.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

Use image attachments for visual description and question answering.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
