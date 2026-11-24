# Raon-1 Mini

Raon-1 Mini is a text model for chat, reasoning, and tool use. The original
checkpoint loads directly and contains about 32.7 GB of weights. Allow extra
memory for inference and prompt caches.
[KRAFTON checkpoint](https://huggingface.co/KRAFTON/Raon-1-mini-preview)

## CLI

```sh
swift run nativ generate \
  --model KRAFTON/Raon-1-mini-preview \
  --prompt "Explain binary search with a short example." \
  --temperature 1.0 --top-p 0.95 --top-k 20 --max-tokens 2048
```

## Swift

```swift
import NativSDK

let model = try await NativModel.loadModel(
    from: .repository("KRAFTON/Raon-1-mini-preview"))
let request = model.generate(
    input: UserInput(
        chat: [.user("What is 17 + 25? Reply with only the number.")],
        additionalContext: ["enable_thinking": false]),
    options: .init(
        sampling: .init(temperature: 0),
        output: .init(maxTokens: 128)))
print(try await request.response())
```

The checkpoint's chat template enables thinking by default. Set
`enable_thinking` to `false` for direct answers, or iterate over the request
to receive reasoning and tool calls as separate events. Pass tool definitions
through `UserInput.tools`. See the [runtime guide](../../Runtime.md) for examples.

## Context and caches

Use the default full-precision cache strategy.

## License

Raon-1 Mini is released under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
See the [model card](https://huggingface.co/KRAFTON/Raon-1-mini-preview) and the
[Python MLX port](https://github.com/Neywa-Labs/mlx-vlm/tree/main/mlx_vlm/models/raon1_mini).
