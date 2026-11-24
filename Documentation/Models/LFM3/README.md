# LFM3-Audio and LFM3-Omni

LFM3-Audio accepts text and audio. LFM3-Omni also accepts images and video.
Both produce text or 24 kHz speech.

Supported checkpoint layouts include the LiquidAI and nativ-community MLX exports:

- `LiquidAI/LFM3-Audio-2B-MLX-4bit`
- `LiquidAI/LFM3-Omni-2B-MLX-4bit`
- `nativ-community/LFM3-Audio-2B-4bit`
- `nativ-community/LFM3-Omni-2B-4bit`

## CLI

Transcribe an audio file:

```sh
swift run -c release nativ generate \
  --model nativ-community/LFM3-Audio-2B-4bit \
  --audio ./speech.wav --prompt "Transcribe the audio." --max-tokens 256
```

Ask about an image (use `--video ./clip.mp4` for video):

```sh
swift run -c release nativ generate \
  --model nativ-community/LFM3-Omni-2B-4bit \
  --image ./image.png --prompt "Describe this image." --max-tokens 256
```

Generate text and speech:

```sh
swift run -c release nativ generate \
  --model nativ-community/LFM3-Audio-2B-4bit \
  --prompt "Say hello and introduce yourself briefly." \
  --audio-mode interleaved --output-audio ./answer.wav --max-tokens 512
```

`--audio-mode` accepts `automatic` (the model chooses), `text`, `audio`, or
`interleaved`. The token limit counts both text tokens and audio frames. Speech
sampling uses checkpoint-family defaults; override with `--audio-temperature`,
`--audio-top-k`, or `--audio-repetition-penalty`. `--seed` seeds both text and
speech sampling. Qwen's `--speaker` option does not select LFM3 voices.

## Swift

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(
    from: .repository("nativ-community/LFM3-Omni-2B-4bit")
)
let input = UserInput(chat: [
    .user("Describe this image aloud.", images: [.url(URL(fileURLWithPath: "image.png"))])
])
let result = try await model.generateAudio(
    input: input,
    options: .init(output: .init(maxTokens: 512)),
    audio: LFM3AudioOptions(mode: .interleaved)
).result()
print(result.text)
// Play or save result.samples using result.sampleRate (24,000 Hz).
```

For text output, use `model.generate(input:).response()`. Audio input is resampled
to 16 kHz mono.

Speech output requires the checkpoint's `audio_tokenizer` directory (the legacy
`audio_detokenizer` name is also recognized). Repository loading downloads these
assets automatically. Include them when copying a checkpoint locally for speech
output.

Generation supports one sequence per request. Supply previous speech as user
audio. Assistant audio history is unsupported. The request returns the complete
waveform once generation finishes.

Prefer `interleaved` for spoken text. Spoken numbers can be inaccurate.

The Audio checkpoint enables thinking by default. For concise text responses,
pass `additionalContext: ["enable_thinking": false]` when constructing `UserInput`.

The native implementation follows [mlx-vlm's LFM3 modules](https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/lfm3_audio),
adapted under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).

## d1 decision models

Load `LiquidAI/d1-2B-MLX-8bit` and `LiquidAI/d1-Omni-2B-MLX-8bit` with
`NativDecisionModel` to get typed decisions.

```swift
let decision = try await NativDecisionModel.loadModel(
    from: .repository("LiquidAI/d1-Omni-2B-MLX-8bit"))
var options = DecisionPredictionOptions()
options.modelOptions = LFM3DecisionOptions(setSize: 16, scoreReadout: .median)
let result = try await decision.predict(
    state: "Check the pictured product and the customer's recorded request.",
    questions: ["refund": .bool("Does the customer request a refund?")],
    options: options,
    media: [.image(imageURL), .audio(recordingURL)]
).result()
print(result.answers["refund"]!.probability!)
```

Both models accept text, JSON, images and video; audio requires d1-Omni. Media
are ordered file URLs. Use `.json(.null)` for media without a text state.
Supported questions are `choice`, `bool`/`noul`, `score` (2–10 ordered levels),
and `multi_label`. A multi-label question evaluates each option independently:

```swift
let question = DecisionQuestion(
    type: .multiLabel, instructions: "Which problems are mentioned?",
    criteria: ["billing", "delivery", "damage"])
```

Boolean answers expose P(yes) in `probability`; choice and multi-label answers
expose their distributions in `probabilities`. Scores default to the calibrated
median. Set `scoreReadout` to `mean` or `argmax` to change it. `setSize` defaults
to 16, and 1 evaluates each question alone. Keep the state within `maxStateTokens`.

```sh
nativ predict --model LiquidAI/d1-Omni-2B-MLX-8bit \
  --audio request.wav --image product.jpg --questions-file questions.json \
  --set-size 16 --score-readout median
```

Supply audio as state evidence alongside written questions. Span extraction,
numeric grids, bounding boxes, and spoken question instructions are unsupported.

If an older MPEG-4 Part 2 (`mp4v`) clip produces green frames, transcode it to
H.264 before inference.

The d1 prompt and readout adaptation follows LiquidAI's System One code under
the [LFM Open License v1.0](D1-LICENSE).
