# Jina VLM

Jina VLM is a vision-language model for image understanding.

- Example checkpoint: `jinaai/jina-vlm-mlx`

## CLI

```sh
swift run nativ generate \
  --model jinaai/jina-vlm-mlx \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 160
```

## Swift

```swift
let source = ModelSource.repository("jinaai/jina-vlm-mlx")
let input = UserInput(chat: [
    .user("Describe this image.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

- Use image prompts for visual description and question answering.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
