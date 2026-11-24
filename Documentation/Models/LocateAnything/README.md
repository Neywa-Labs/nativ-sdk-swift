# Locate Anything

Locate Anything is a grounding-oriented vision-language family for object
localization prompts.

- Example checkpoint: `mlx-community/LocateAnything-3B-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/LocateAnything-3B-4bit \
  --image ./image.jpg \
  --prompt "Locate the main objects." \
  --max-tokens 256
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/LocateAnything-3B-4bit")
let input = UserInput(chat: [
    .user("Locate the main objects.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

- Use grounding prompts with image attachments.
- Outputs should be interpreted according to the checkpoint's grounding format.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
