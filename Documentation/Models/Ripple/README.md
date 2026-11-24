# Ripple-65B-Preview

Ripple is a text model for coding, reasoning, and tool use. Its decoder combines
1,024-token sliding attention with global attention every fifth layer. Sliding
layers use rotary position embeddings, and global layers retain the full context.

## Checkpoints

- [ricursiveAI/Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview)
  provides the original BF16 weights, about 129.6 GB.
- [frogggias/Ripple-65B-Preview-MLX-4bit](https://huggingface.co/frogggias/Ripple-65B-Preview-MLX-4bit)
  is a community MLX affine 4-bit export, about 36.4 GB.

Allow additional memory for prompt caches, runtime allocations, and other
applications. The SDK loads safetensors directly and uses the checkpoint's
tokenizer and chat template without executing Python code.

## CLI

```sh
swift run nativ generate \
  --model frogggias/Ripple-65B-Preview-MLX-4bit \
  --prompt "Write a Python function that merges overlapping intervals." \
  --temperature 1.0 --top-p 0.95 --max-tokens 2048
```

## Swift

```swift
import NativSDK

let model = try await NativModel.loadModel(
    from: .repository("frogggias/Ripple-65B-Preview-MLX-4bit"))
let request = model.generate(
    input: UserInput(chat: [.user("Explain how to merge overlapping intervals.")]),
    options: .init(
        sampling: .init(temperature: 1.0, topP: 0.95),
        output: .init(maxTokens: 2048)))
print(try await request.response())
```

The token budget covers both reasoning and answer text. Iterate over the request
to receive reasoning and tool-call events. When supplying raw assistant history,
preserve `reasoning_content` and pass tool arguments as JSON objects. See the
[runtime guide](../../Runtime.md) for streaming and tool definitions.

Ripple uses ordinary autoregressive generation in the SDK. The DFlash2 drafter
requires separate support.

## License

Ripple is released under [Apache 2.0](https://huggingface.co/ricursiveAI/Ripple-65B-Preview/blob/main/LICENSE).
The [upstream implementation](https://huggingface.co/ricursiveAI/Ripple-65B-Preview/blob/main/modeling_ricursive.py)
and [Python MLX port](https://github.com/Neywa-Labs/mlx-vlm/tree/main/mlx_vlm/models/ricursive)
describe the architecture.
