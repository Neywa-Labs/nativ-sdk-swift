# Qwen3 Omni

Qwen3 Omni accepts text, images, and audio, and can produce text or speech responses.

- Example checkpoint: `mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit`

## CLI

Text or vision:

```sh
swift run nativ generate \
  --model mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

Audio output:

```sh
swift run nativ generate \
  --model mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit \
  --audio ./clip.wav \
  --prompt "Answer out loud." \
  --output-audio ./answer.wav \
  --max-tokens 256
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit")
let input = UserInput(chat: [
    .user("Describe this audio.", audios: [.url(URL(fileURLWithPath: "clip.wav"))])
])

try await generateText(source: source, input: input)
```

Audio waveform generation:

```swift
let model = try await NativModel.loadModel(
    from: source
)

let prepared = try await model.prepare(input: input)
let result = try await model.generateAudio(
    prepared: prepared,
    options: .init(output: .init(maxTokens: 256))
).result()
print(result.shape)
```

## Notes

Speech output supports one response at a time, uses greedy sampling, and produces 24 kHz audio.
The CLI writes WAV files at the model's native sample rate; it does not resample audio.
Swift callers should use `result.sampleRate` when playing or saving `result.samples`.
Other model families may support audio input without speech output.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
