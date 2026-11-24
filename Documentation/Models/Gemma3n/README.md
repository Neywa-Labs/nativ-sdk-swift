# Gemma 3n

Gemma 3n is a multimodal Gemma family supporting text, image, and audio inputs.

- Example checkpoint: `mlx-community/gemma-3n-E2B-it-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/gemma-3n-E2B-it-4bit \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 160
```

Audio:

```sh
swift run nativ generate \
  --model mlx-community/gemma-3n-E2B-it-4bit \
  --audio ./clip.wav \
  --prompt "Transcribe this audio." \
  --max-tokens 160
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/gemma-3n-E2B-it-4bit")
let input = UserInput(chat: [
    .user("Transcribe this audio.", audios: [.url(URL(fileURLWithPath: "clip.wav"))])
])

try await generateText(source: source, input: input)
```

## Notes

- Image and audio inputs are supported for compatible Gemma 3n checkpoints.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
