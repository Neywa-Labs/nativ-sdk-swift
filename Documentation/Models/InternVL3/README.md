# InternVL3

InternVL3 supports image conversations and visual question answering.

- Example checkpoint: `mlx-community/InternVL3-1B-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/InternVL3-1B-4bit \
  --image ./image.jpg \
  --prompt "Describe this image in detail." \
  --max-tokens 160
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/InternVL3-1B-4bit")
let input = UserInput(chat: [
    .user("Describe this image in detail.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

- Prompts may include an explicit `<image>` marker, but it is not required.
- Large images are tiled dynamically.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
