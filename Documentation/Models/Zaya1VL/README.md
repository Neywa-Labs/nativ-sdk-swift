# ZAYA1-VL

ZAYA1-VL supports conversations about images.

- Example checkpoint: `OsaurusAI/ZAYA1-VL-8B-MXFP4`

## CLI

```sh
swift run nativ generate \
  --model OsaurusAI/ZAYA1-VL-8B-MXFP4 \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 160
```

## Swift

```swift
let source = ModelSource.repository("OsaurusAI/ZAYA1-VL-8B-MXFP4")
let input = UserInput(chat: [
    .user("Describe this image.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

try await generateText(source: source, input: input)
```

## Notes

Use chat prompts with image attachments.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
