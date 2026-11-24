# Llama 4 Scout

Llama 4 Scout supports conversations about text and images.

- Example checkpoint: `mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit")
let input = UserInput(chat: [
    .user("Describe this image.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

- Use Llama 4 multimodal prompts with image attachments.
- Direct repo IDs can be used for compatible MLX Llama 4 checkpoints.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
