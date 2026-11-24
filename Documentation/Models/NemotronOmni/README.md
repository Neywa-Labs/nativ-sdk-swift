# Nemotron Omni

Nemotron Omni supports audio prompts with text responses.

## CLI

```sh
swift run nativ generate \
  --model nvidia/Nemotron-H-Nano-9B-v2-Omni \
  --audio ./clip.wav \
  --prompt "Transcribe this audio." \
  --max-tokens 256
```

## Swift

```swift
let source = ModelSource.repository("nvidia/Nemotron-H-Nano-9B-v2-Omni")
let input = UserInput(prompt: "Transcribe this audio.", audios: [
    .url(URL(fileURLWithPath: "clip.wav"))
])

try await generateText(source: source, input: input)
```

## Notes

Audio input with text responses is supported. Image and video input are unsupported.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
