# Ling 3.1 Flash

Ling 3.1 Flash is a text model with Kimi Delta Attention, gated multi-head
latent attention, and shared and routed experts. The model has 560 billion
parameters and activates about 25 billion per token. See the
[inclusionAI model card](https://huggingface.co/inclusionAI/Ling-3.1-flash).

Use a compatible MLX checkpoint that fits in memory, with room for caches and
runtime allocations. The original checkpoint contains about 573.8 GB of
safetensors. Loading those source weights expands block FP8 tensors to BF16.

## CLI

```sh
swift run nativ generate \
  --model ./Ling-3.1-flash-4bit \
  --prompt "Explain why the sky is blue." \
  --temperature 1.0 --top-p 0.95 --top-k 20 --max-tokens 2048
```

## Swift

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(
    from: .directory(URL(fileURLWithPath: "./Ling-3.1-flash-4bit")))
let request = model.generate(
    input: UserInput(
        chat: [.user("Give a short example of binary search.")],
        additionalContext: ["enable_thinking": false]),
    options: .init(
        sampling: .init(temperature: 1.0, topP: 0.95, topK: 20),
        output: .init(maxTokens: 2048)))
print(try await request.response())
```

The checkpoint's chat template enables thinking by default. Set
`enable_thinking` to `false` for direct answers. Iterate over the request to
receive reasoning and tool call events. Pass available tools through
`UserInput.tools`. See the [runtime guide](../../Runtime.md).

The SDK supports autoregressive generation with FP32 recurrent state and
full-precision latent KV caches. Use the default cache configuration.

## License

Ling 3.1 Flash is released under the MIT license. See the
[published checkpoint](https://huggingface.co/inclusionAI/Ling-3.1-flash)
and the [Python MLX implementation](https://github.com/Neywa-Labs/mlx-vlm/tree/main/mlx_vlm/models/bailing_hybrid).
