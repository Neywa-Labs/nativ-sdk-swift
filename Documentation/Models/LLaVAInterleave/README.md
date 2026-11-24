# LLaVA Interleave Qwen

LLaVA Interleave Qwen is a compact LLaVA-style vision-language model.

- Example checkpoint: `mlx-community/llava-interleave-qwen-0.5b-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/llava-interleave-qwen-0.5b-4bit \
  --image ./image.jpg \
  --prompt "Describe this image in detail." \
  --max-tokens 128
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/llava-interleave-qwen-0.5b-4bit")
let input = UserInput(chat: [
    .user("Describe this image in detail.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

Use image prompts for description and visual question answering.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
