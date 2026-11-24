# Falcon Perception

Falcon Perception is a compact perception and grounding model for object
description, detection, and localization-style prompts.

- Example checkpoint: `tiiuae/Falcon-Perception-300M`

## CLI

```sh
swift run nativ generate \
  --model tiiuae/Falcon-Perception-300M \
  --image ./image.jpg \
  --prompt "the main objects" \
  --max-tokens 256
```

## Swift

```swift
let source = ModelSource.repository("tiiuae/Falcon-Perception-300M")
let input = UserInput(chat: [
    .user("the main objects", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

- Perception prompts can produce object descriptions and localization-oriented
  responses.
- Segmentation tasks are not exposed through text generation yet.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
