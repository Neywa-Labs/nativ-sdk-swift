# Youtu VL

Youtu VL supports image understanding and visual question answering.

- Example checkpoint: `tencent/Youtu-VL-4B-Instruct`

## CLI

```sh
swift run nativ generate \
  --model tencent/Youtu-VL-4B-Instruct \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 160
```

## Swift

```swift
let source = ModelSource.repository("tencent/Youtu-VL-4B-Instruct")
let input = UserInput(chat: [
    .user("Describe this image.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

Use image prompts for description and visual question answering.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
