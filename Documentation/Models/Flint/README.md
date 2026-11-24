# Flint

Flint is a text model for reasoning, coding, and tool use, with 39.8 billion
parameters and approximately 2.5 billion active parameters per token.

## Model

- [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint)
- [mlx-community/flint-oQ6e](https://huggingface.co/mlx-community/flint-oQ6e),
  a mixed 6-bit/8-bit MLX checkpoint of approximately 32.8 GB.

The original BF16 safetensors load directly and occupy approximately 79.6 GB.
Allow additional memory for caches, runtime allocations, and other applications.
Converted MLX checkpoints, including mixed-bit exports, are also supported.

## CLI

```sh
swift run nativ generate \
  --model mlx-community/flint-oQ6e \
  --prompt "Write a Python function that merges overlapping intervals." \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --max-tokens 4096
```

## Swift

```swift
import NativSDK
import MLX

let ticket = WiredMemoryTicket(
    size: GPU.maxRecommendedWorkingSetBytes() ?? 0, policy: WiredMaxPolicy())
let model = try await ticket.withWiredLimit {
    try await NativModel.loadModel(
        from: .repository("RecursiveSI/flint"),
        policy: .init(wiredMemoryTicket: ticket))
}
let request = model.generate(
    input: UserInput(chat: [.user("Explain binary search and give a Python example.")]),
    options: .init(
        sampling: .init(temperature: 1.0, topP: 0.95, topK: 20),
        output: .init(maxTokens: 4096)))
print(try await request.response())
```

For the large BF16 checkpoint, the ticket keeps model memory resident during
loading, and the execution policy does the same during generation.
Use a budget that fits the device and other active workloads;
the example uses Metal's recommended working-set size.

Flint can generate reasoning before its answer. The token budget includes both.
Iterate over the request to receive reasoning and tool-call events alongside
response text; see the [runtime guide](../../Runtime.md).

## Conversation history and tools

The checkpoint's chat template formats messages and tool definitions. When
supplying raw message history through `UserInput(prompt: .messages(...))`, keep
assistant `reasoning_content` fields and `tool_calls`, followed by the matching
tool responses. Function arguments in history may be dictionaries or JSON strings.
Pass available function definitions through `UserInput.tools`.

## Context and caches

The checkpoint configures a maximum context of 262,144 tokens. Actual context
capacity depends on available device memory. Use a full-precision or affine
quantized cache. TurboQuant is unsupported.

## License and references

Flint uses the [Apache 2.0 license](https://huggingface.co/RecursiveSI/flint/blob/main/LICENSE).
See the [model card](https://huggingface.co/RecursiveSI/flint) and the
[Python MLX implementation](https://github.com/Neywa-Labs/mlx-vlm/tree/main/mlx_vlm/models/flint).
