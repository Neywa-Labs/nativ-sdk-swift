# Llama 3.2 Vision

Llama 3.2 Vision is Meta's multimodal Llama family for image understanding.

- Example checkpoint: `mlx-community/Llama-3.2-11B-Vision-Instruct-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/Llama-3.2-11B-Vision-Instruct-4bit \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 160
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/Llama-3.2-11B-Vision-Instruct-4bit")
let input = UserInput(chat: [
    .user("Describe this image.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

Use image attachments for visual question answering.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
