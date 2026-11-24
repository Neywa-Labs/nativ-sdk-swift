# MolmoPoint

MolmoPoint is a pointing and localization-oriented Molmo variant.

- Example checkpoint: `mlx-community/MolmoPoint-8B-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/MolmoPoint-8B-4bit \
  --image ./image.jpg \
  --prompt "Point to the main object." \
  --max-tokens 160
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/MolmoPoint-8B-4bit")
let input = UserInput(chat: [
    .user("Point to the main object.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

- Use image prompts for MolmoPoint low-resolution grounding tasks.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
