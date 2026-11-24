# Nemotron 3.5 Transcribe

Nemotron 3.5 Transcribe transcribes and translates recordings, attributes
speech to speakers, and answers questions about audio.
[Source checkpoint](https://huggingface.co/nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-bf16)

## Prepare for mobile

Convert the native MLX checkpoint on a Mac:

```sh
swift run -c release nativ convert /path/to/bf16 /path/to/4bit --bits 4 --group-size 64
```

The default conversion policy quantizes eligible language weights and keeps
the speech and speaker encoders at source precision.

Use the optional mixed recipe to reduce some expert weights to 3-bit:

```sh
swift run -c release nativ convert /path/to/bf16 /path/to/3bit-experts \
  --bits 4 --group-size 64 --quant-predicate mixed
```

Start with the floating-point source checkpoint. Conversion preserves weights
that are already quantized.

Enable expert streaming to lower the memory required for expert weights and
audio encoding. Allow memory beyond the configured expert cache for the rest of
the model and audio processing. [Expert streaming guide](../../ExpertStreaming.md)

## Swift

```swift
let model = try await NativModel.loadModel(
    from: .directory(URL(fileURLWithPath: "/path/to/4bit")),
    expertStreaming: .init(
        cacheSizeBytes: 1024 * 1024 * 1024, sharedExpertBuffers: true))
let request = model.generate(
    input: UserInput(
        prompt: "", audios: [.url(URL(fileURLWithPath: "/path/to/clip.wav"))],
        additionalContext: ["task": "translate", "language": "German"]),
    options: .init(prefill: .init(stepSize: 256), sampling: .init(temperature: 0),
                   output: .init(maxTokens: 512)))
for try await event in request {
    if case .chunk(let text) = event { print(text, terminator: "") }
}
```

| Task | Context | Prompt |
| --- | --- | --- |
| Transcription | `task: asr` (default) | Empty selects native transcription instructions. |
| Translation | `task: translate`, `language: German` | Empty selects native translation instructions. |
| Speaker attribution | `task: saa` | Empty selects instructions preserving `<spk:N>` tags. |
| Audio Q&A | `task: qa` | A question about the recording, or empty for audio-only input. |
| Rich transcription | `task: rich` | Empty requests native event/style markup. |

A nonempty prompt overrides the task's default instructions. Optional
`hotwords: [String]` adds contextual vocabulary. `enable_thinking` defaults to
`false`. Send one recording per request. Include a system message to supply
optional voice-assistant instructions.

Files are decoded and resampled to mono 16 kHz. Array inputs must already be mono at 16 kHz. Output text streams after audio encoding. Speaker labels do not identify real people, and word timestamps are unavailable. Rich tags and speaker assignments depend on model accuracy.

See the [iPhone and iPad example](../../../Examples/NemotronTranscribeLab/README.md)
for file import, microphone recording, and bundled samples.
