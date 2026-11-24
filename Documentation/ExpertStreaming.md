# Expert Streaming

Disk-backed expert streaming reduces memory use by reading routed expert
weights from disk when needed. Use a compatible local mixture-of-experts (MoE)
checkpoint.

```swift
let model = try await NativModel.loadModel(
    from: .directory(modelDirectory),
    expertStreaming: .init(cacheSizeBytes: 64 * 1024 * 1024))
```

```sh
swift run -c release nativ generate --model ./model \
  --prompt "Explain how a rainbow forms." --max-tokens 128 \
  --expert-cache-mb 64 --prefill-step-size 256
```

## Memory and generation options

- `cacheSizeBytes` (CLI `--expert-cache-mb`) sets the expert cache budget. Allow
  extra memory for resident model weights, media processing, the conversation
  cache, and temporary allocations. The cache must fit at least one expert
  projection, or one whole expert for the MiniCPM-V 4.7 and Gemma 4 adapters.
- `expertBatchSize` (CLI `--expert-batch-size`) limits how many experts are evaluated
  together. The default is 8. Memory limits can reduce the effective batch size.
- Expert prefetching is on by default. Set `prefetch: false` to load experts on demand.
- `prefill.stepSize` (CLI `--prefill-step-size`) limits the number of prompt tokens
  processed per chunk. It defaults to 256 on iOS and iPadOS and 512 elsewhere.
  Smaller chunks reduce temporary memory use.

Keep checkpoint files available and unchanged while the model is loaded.
Expert streaming works with autoregressive and diffusion text generation. In
the CLI, use it with text output and without a draft model. To train or save
a model, load it without expert streaming.

MiniCPM-V 4.7 and Gemma 4 use router lookahead for expert prefetching. The
generic MoE path uses recent routing history. The model's router still selects
which experts run. Router lookahead follows the approach described in
[OpenBMB's Lookahead Expert Scheduling (LES) proposal](https://github.com/OpenBMB/llama.cpp-omni/blob/main/docs/les/lookahead-expert-scheduling.md).

## Shared expert buffers

In Swift, set `sharedExpertBuffers: true` to enable shared buffers on checkpoints
that support them, including MiniCPM-V 4.7 and Gemma 4. The model uses the
standard expert cache when the budget is too small for shared buffers. On
MiniCPM-V and Gemma 4, set `deferVisionWeights: true` to keep vision weights out
of memory until the first image prompt. Both options default to `false`.

```swift
let model = try await NativModel.loadModel(
    from: .directory(modelDirectory),
    expertStreaming: .init(
        cacheSizeBytes: 1408 * 1024 * 1024,
        sharedExpertBuffers: true,
        deferVisionWeights: true))
```

## Checkpoint layout

Use safetensors in the model's native MLX layout with stacked expert weights.
Floating-point weights and MLX affine, MXFP4, MXFP8, and NVFP4 quantization are
supported. A checkpoint that needs weight conversion must be prepared before
loading with expert streaming.

The following source checkpoints and layouts can also load directly:

| Model | Supported source layout |
| --- | --- |
| LFM2 MoE | Separate floating-point Transformers experts, including the BF16 LFM2.5 24B A2B Preview |
| [MiniCPM-V 4.7](Models/MiniCPMV/README.md#minicpm-v-47-35b-a3b-with-expert-streaming) | Original fused Transformers experts or separate expert tensors |
| Gemma 4 MoE | Original fused floating-point experts |
| `CohereLabs/North-Mini-Code-1.0-w4a16` | Official W4A16 compressed-tensors checkpoint |
| `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-bf16` | Official BF16 checkpoint |
| `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` | Official mixed FP8/NVFP4 ModelOpt checkpoint |

The North Mini Code checkpoint also loads without expert streaming. Use the
original checkpoint directory for Nemotron 3.5 Bolt and Super-VL. These models
still have substantial memory requirements with expert streaming enabled.

## Preparing a smaller checkpoint

Expert streaming uses the checkpoint's weight precision. To reduce the size
of resident weights, use a compatible quantized checkpoint or convert a native
MLX checkpoint with `nativ convert`. For model-specific conversion commands,
see [MiniCPM-V](Models/MiniCPMV/README.md#prepare-weights-on-a-mac),
[Nemotron Transcribe](Models/NemotronTranscribe/README.md#prepare-for-mobile),
and the [CLI guide](../Sources/Tools/nativ/README.md).

For LFM2 MoE, the default conversion keeps routers at 8-bit and uses the
requested precision for other eligible weights. Select `--quant-predicate mixed`
to use MXFP4 routed experts, 8-bit routers, and the requested precision for
eligible resident weights. Existing quantized weights keep their precision, so
you can also use the recipe to compress the resident weights of a checkpoint
whose experts are already quantized:

```sh
nativ convert ./LFM2.5-MXFP4-Experts ./LFM2.5-Mixed --quant-predicate mixed --bits 4 --group-size 64
```

Convert from native MLX weights, even if the original Transformers checkpoint
can load directly with expert streaming.
