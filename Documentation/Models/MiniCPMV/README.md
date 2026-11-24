# MiniCPM-V 4.6 and 4.7

MiniCPM-V 4.6 is a vision-language model with image and video input support.

- Example checkpoint: `mlx-community/MiniCPM-V-4.6-4bit`

## CLI

Image:

```sh
swift run nativ generate \
  --model mlx-community/MiniCPM-V-4.6-4bit \
  --image ./image.jpg \
  --prompt "Describe this image in detail." \
  --max-tokens 160
```

Video:

```sh
swift run nativ generate \
  --model mlx-community/MiniCPM-V-4.6-4bit \
  --video ./clip.mp4 \
  --prompt "Summarize this video." \
  --max-tokens 256
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/MiniCPM-V-4.6-4bit")
let input = UserInput(chat: [
    .user("Summarize this video.", videos: [.url(URL(fileURLWithPath: "clip.mp4"))])
])

try await generateText(source: source, input: input)
```

## Notes

Image and video inputs are supported. Image sizing is handled automatically.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.

<a name="minicpm-v-47-35b-a3b-with-les"></a>

## MiniCPM-V 4.7 35B-A3B with expert streaming

MiniCPM-V 4.7 accepts image and video prompts and defaults to instruct mode. Pass
`additionalContext: ["enable_thinking": true]` on `UserInput` to enable thinking.

Enable expert streaming when loading to read routed expert weights from disk.
[Expert streaming guide](../../ExpertStreaming.md)

```swift
import MLX
import NativSDK

// This is MLX's allocator cache, separate from the expert cache below.
Memory.cacheLimit = 64 * 1024 * 1024
let model = try await NativModel.loadModel(
    from: .directory(modelDirectory),
    expertStreaming: .init(cacheSizeBytes: 64 * 1024 * 1024)
)
let response = try await model.generate(
    input: UserInput(chat: [.user("Describe this image.", images: [.url(imageURL)])]),
    options: .init(output: .init(maxTokens: 128))
).response()
print(response)
```

The expert cache budget covers routed weights. Allow additional memory for
resident weights, media processing, and the conversation cache. Smaller
prefill chunks reduce temporary memory use. Longer conversations and larger
images increase memory use.

### Prepare weights on a Mac

For iPhone, use the community MLX checkpoint and quantize its resident weights
on an Apple silicon Mac before transferring the completed directory:

```sh
hf download mlx-community/MiniCPM-V-4.7-35B-A3B-mxfp4 \
  --local-dir ./MiniCPM-V-4.7-mxfp4

swift run -c release nativ convert \
  ./MiniCPM-V-4.7-mxfp4 ./MiniCPM-V-4.7-streaming --quant-predicate mixed

swift run nativ generate \
  --model ./MiniCPM-V-4.7-streaming \
  --expert-cache-mb 64 --cache-limit-mb 64 \
  --image ./image.jpg --prompt "Describe this image." \
  --max-tokens 128
```

The `mixed` recipe preserves routed experts and already quantized weights.
It defaults to 4-bit for eligible resident language weights and uses 8-bit for
routers and eligible vision weights. Add `--bits 8` for higher-precision resident
language weights.

The checkpoint prepared above occupies about **19.2 GB of storage**. Conversion
requires MLX safetensors, a new destination directory, and extra memory and disk
space while it runs.
