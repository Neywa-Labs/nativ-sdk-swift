# Phi-4 Multimodal

Phi-4 Multimodal supports audio transcription and questions about audio in Nativ SDK.

## CLI

```sh
swift run nativ generate \
  --model microsoft/Phi-4-multimodal-instruct \
  --audio ./clip.wav \
  --prompt "Transcribe this audio." \
  --max-tokens 256
```

## Swift

```swift
let source = ModelSource.repository("microsoft/Phi-4-multimodal-instruct")
let input = UserInput(prompt: "Transcribe this audio.", audios: [
    .url(URL(fileURLWithPath: "clip.wav"))
])

try await generateText(source: source, input: input)
```

## Notes

Audio input is supported for transcription and questions about recordings. Image input is unsupported.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
