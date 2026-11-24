# Idefics2

Idefics2 is a vision-language model for image understanding and VQA.

- Example checkpoint: `mlx-community/idefics2-8b-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/idefics2-8b-4bit \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 160
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/idefics2-8b-4bit")
let input = UserInput(chat: [
    .user("Describe this image.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

Use chat prompts with image attachments.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
