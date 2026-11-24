# Moondream3

Moondream3 is the preview Moondream vision-language family.

- Example checkpoint: `beshkenadze/moondream3-preview-mlx-4bit`
- Tokenizer source: `moondream/starmie-v1`

## CLI

```sh
swift run nativ generate \
  --model beshkenadze/moondream3-preview-mlx-4bit \
  --image ./image.jpg \
  --prompt "Describe this image in detail." \
  --max-tokens 160
```

## Swift

```swift
let source = ModelSource.repository("beshkenadze/moondream3-preview-mlx-4bit")
let input = UserInput(chat: [
    .user("Describe this image in detail.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

- The preview MLX checkpoint uses the separate `moondream/starmie-v1` tokenizer source.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
