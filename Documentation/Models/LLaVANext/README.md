# LLaVA Next Mistral

LLaVA Next supports image description and visual question answering.

- Example checkpoint: `mlx-community/llava-v1.6-mistral-7b-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/llava-v1.6-mistral-7b-4bit \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 160
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/llava-v1.6-mistral-7b-4bit")
let input = UserInput(chat: [
    .user("Describe this image.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

- Use image prompts for visual description and question answering.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
