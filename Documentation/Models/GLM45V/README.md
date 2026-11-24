# GLM 4.5V

GLM 4.5V supports image understanding and visual reasoning.

- Example checkpoint: `mlx-community/GLM-4.5V-3bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/GLM-4.5V-3bit \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/GLM-4.5V-3bit")
let input = UserInput(chat: [
    .user("Describe this image.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

Use image prompts for description and visual reasoning.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
