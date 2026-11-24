# MiniCPM-o

MiniCPM-o is an omni model with text, image, and audio input support.

- Example checkpoint: `mlx-community/MiniCPM-o-4_5-4bit`

## CLI

Image:

```sh
swift run nativ generate \
  --model mlx-community/MiniCPM-o-4_5-4bit \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 160
```

Audio:

```sh
swift run nativ generate \
  --model mlx-community/MiniCPM-o-4_5-4bit \
  --audio ./clip.wav \
  --prompt "Describe this audio briefly." \
  --max-tokens 256
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/MiniCPM-o-4_5-4bit")
let input = UserInput(chat: [
    .user(
        "Describe what you see and hear.",
        images: [.url(URL(fileURLWithPath: "image.jpg"))],
        audios: [.url(URL(fileURLWithPath: "clip.wav"))]
    )
])

try await generateText(source: source, input: input)
```

## Notes

- Prompts can include `<image>` and `<audio>` placeholders.
- Attach audio with `UserInput.audios`.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
