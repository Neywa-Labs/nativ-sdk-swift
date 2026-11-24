# Step-3.7 Flash

Step-3.7 Flash is a vision-language model for image understanding.

- Example checkpoint: `ivanfioravanti/Step-3.7-Flash-4bit`

## CLI

```sh
swift run nativ generate \
  --model ivanfioravanti/Step-3.7-Flash-4bit \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 160
```

## Swift

```swift
let source = ModelSource.repository("ivanfioravanti/Step-3.7-Flash-4bit")
let input = UserInput(chat: [
    .user("Describe this image.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

- Use image prompts for visual description and reasoning tasks.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
