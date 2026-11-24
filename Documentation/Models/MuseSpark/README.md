# Muse Spark Open

Built with Muse Spark Open.

Muse Spark Open generates text from text, images, video frames, and audio.
Pass media together in one request when the checkpoint includes the corresponding
encoders. The [MLX community oQ4e checkpoint](https://huggingface.co/mlx-community/Muse-Spark-Open-816B-A42B-oQ4e)
provides text, image, and video support. For audio, use a checkpoint with
the waveform tower, such as [Meta’s NVFP4 release](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4).

The release has 816 billion parameters and activates about 42 billion per
token. Choose a checkpoint that fits in the runtime’s available memory, including
its caches. [Meta’s model download guide](https://www.musemuse.dev/docs/muse-spark-open/get-the-model)

## CLI

```sh
swift run nativ generate --model ./Muse-Spark-Open \
  --image photo.jpg --video clip.mp4 --audio recording.wav \
  --prompt "Compare the image, video, and recording." --max-tokens 512
```

Use the media flags that match your checkpoint. Video input supplies visual
frames with timestamps. Include an audio input to ask about a soundtrack.

## Swift

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(
    from: .directory(URL(fileURLWithPath: "./Muse-Spark-Open")))
let input = UserInput(
    chat: [.user(
        "Compare the image and recording.",
        images: [.url(URL(fileURLWithPath: "photo.jpg"))],
        audios: [.url(URL(fileURLWithPath: "recording.wav"))])],
    additionalContext: ["reasoning_effort": "low"])
let request = model.generate(input: input, options: .init(output: .init(maxTokens: 512)))
print(try await request.response())
```

The audio encoder expects mono samples at 24 kHz. Pass in-memory audio as
`.array(samples, sampleRate: sourceRate)` to resample it to 24 kHz, or use
`.array(samples)` for a prepared 24 kHz waveform. The released encoder uses one
token per 1,920 samples.
The default video settings sample at 2 fps with up to 600 frames. Set
`UserInput.Processing.video` to customize sampling, or pass decoded frames
with their timestamps.

The response stream supports reasoning and ATEM tool-call events. Set
`reasoning_effort` to `low`, `medium`, `high`, or `xhigh` in the chat template
context to choose the reasoning effort.

## License

The weights are distributed under the [Muse Model License and Usage Policy](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4/blob/main/LICENSE.md).
See the [MLX Python implementation](https://github.com/Neywa-Labs/mlx-vlm/tree/main/mlx_vlm/models/muse_spark).
