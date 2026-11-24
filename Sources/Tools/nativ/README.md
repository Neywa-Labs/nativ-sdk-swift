# nativ CLI

Run local models from the command line. Pass a Hugging Face repository ID or a
local checkpoint directory to `--model`. Models download automatically on first use.

```sh
swift run nativ --help
swift run nativ generate --help
```

## Text and conversations

```sh
swift run nativ generate --model nativ-community/LFM3-Omni-2B-4bit \
  --prompt "In one sentence, why is the sky blue?"
swift run nativ chat --model nativ-community/LFM3-Omni-2B-4bit
```

Use `--prompt-file` for a prompt stored in a file. `--max-tokens` limits response
length; `--temperature`, `--top-p`, `--top-k`, and `--seed` control sampling.
`--revision` selects a model revision. Interactive `chat` is text-only and uses
ordinary generation.

`--prefill-step-size` sets the maximum tokens per prefill chunk for `generate`
and `chat`. The default is 256 on iOS/iPadOS and 512 elsewhere. Larger chunks can
improve prompt processing speed and require more working memory.

## Images, video, and audio

Use media options with a checkpoint that supports the input type:

```sh
swift run nativ generate --model mlx-community/llava-interleave-qwen-0.5b-4bit \
  --image ./image.jpg --prompt "Describe this image."
swift run nativ generate --model mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit \
  --audio ./clip.wav --prompt "Transcribe this."
```

See the [model catalog](../../../Documentation/Models/README.md) for video examples,
audio output, prompt formats, and model-specific limitations.

For MoE models, `generate --expert-cache-mb 64` enables disk-backed expert
streaming. `--expert-batch-size` controls the maximum experts per evaluation (default
8). It requires `--expert-cache-mb`. Use `--prefill-step-size` to adjust prompt chunk size.
Use a checkpoint in the model's native MLX expert layout or a supported
separate-expert layout such as LFM2.5 24B A2B Preview's BF16 Transformers weights.
[Expert streaming guide](../../../Documentation/ExpertStreaming.md)
See the [MiniCPM-V guide](../../../Documentation/Models/MiniCPMV/README.md#minicpm-v-47-35b-a3b-with-expert-streaming)
for preparing weights for iPhone and the Swift loading option. For Gemma 4 MoE
with expert streaming, a quantized checkpoint such as
`mlx-community/gemma-4-26b-a4b-it-4bit` reduces the memory needed for resident
weights. Those weights and other working memory are additional to the expert
cache budget.

Quantize a local MLX checkpoint using its model policy, or select an explicit mixed recipe:

```sh
swift run -c release nativ convert ./source-model ./converted-8bit --bits 8
swift run -c release nativ convert ./source-model ./converted-mixed --quant-predicate mixed
swift run -c release nativ convert ./source-model ./converted-4-8 --quant-predicate mixed_4_8
```

Omit `--quant-predicate` to use the model's default rules for eligible layers.
Set `--bits`, `--group-size`, and `--mode affine|mxfp4|mxfp8|nvfp4` to change the
requested precision. The default is affine 4-bit with group size 64. Non-affine
modes use their required bits and group size when omitted. Vision and audio
weights are excluded by default. Already quantized weights keep their format.

`--quant-predicate mixed` selects a model-specific recipe for
[LFM2 MoE](../../../Documentation/ExpertStreaming.md#preparing-a-smaller-checkpoint),
[MiniCPM-V 4.7](../../../Documentation/Models/MiniCPMV/README.md#prepare-weights-on-a-mac),
or [Nemotron Transcribe](../../../Documentation/Models/NemotronTranscribe/README.md#prepare-for-mobile).

Other recipes are `mixed_2_6`, `mixed_3_4`, `mixed_3_5`, `mixed_3_6`,
`mixed_3_8`, `mixed_4_6`, and `mixed_4_8`. These use two bit widths for eligible
language weights, with affine group size 64. They require indexed decoder layers
with `down_proj` or `feed_forward.w2`. Use them without `--bits`, `--group-size`,
or `--mode`. An explicit recipe overrides the model's default rules.

Conversion runs on a Mac and needs additional memory and disk space. Use a
checkpoint in the model's native MLX safetensors layout and a new destination
directory. In Swift, use `ModelConversion.convert(..., recipe: .mixed)`, another
`QuantizationRecipe` value, or a custom predicate.

## Image generation and editing

```sh
swift run nativ image --model black-forest-labs/FLUX.2-klein-4B \
  --prompt "A small red apple on a wooden table." --output ./apple.png
swift run nativ image --model black-forest-labs/FLUX.2-klein-4B \
  --task edit --image ./reference.png --prompt "Replace the background with a forest." \
  --output ./edit.png
```

`image` and `generate_image` are aliases. Use `--size`, `--seed`, `--steps`, and
`--guidance` to customize output; omit sampling controls to use model defaults.
Editing support and additional options vary by model. See
[image generation](../../../Documentation/ImageGeneration.md).

## Draft models

Add `--draft-model` to use a compatible draft with your target:

```sh
swift run nativ generate --model LiquidAI/LFM2.5-2.6B \
  --draft-model LiquidAI/LFM2.5-2.6B-DSpark --prompt "Write a short Python function."
```

Supported pairs and proposal counts:

| Target and draft | Inputs and sampling | Default / maximum proposals |
| --- | --- | --- |
| Independent Qwen3 or Mistral4 checkpoints with matching tokenizers | Text, greedy | 4 / 16 |
| LFM2.5-2.6B + matching DSpark | Text, greedy or sampled | 7 / 9 |
| LFM2.5-VL-3B + matching DSpark | Text and images, greedy or sampled | 7 / 9 |
| Dense Qwen3 + compatible DFlash / DSpark | Text, greedy or sampled | 7 / 15 for block-16 DFlash; 7 / 7 for block-7 DSpark |
| Qwen3.5 + matching DFlash | Text and images, greedy or sampled | 7 / 15 |
| Qwen3.5-family + matching MTP head | Text, greedy | 1 / 1 |
| Laguna XS 2.1 / S 2.1 + matching DFlash | Text, greedy or sampled | 15 / 15 |
| Gemma 4 + matching DFlash / DSpark | Text and images, greedy or sampled | 7 / 15 for block-16 DFlash; 7 / 7 for block-7 DSpark |

Use `--draft-tokens` to change the proposal count and `--draft-revision` to select
the draft revision. LFM2 pairs require the original, unquantized BF16 targets.
DFlash and DSpark require unbounded, full-precision caches. Speed depends on the
pair, prompt, and sampling settings; a draft is not always faster.
See the [Gemma 4 guide](../../../Documentation/Models/Gemma4Unified/README.md#draft-models)
for compatible checkpoints and generation limitations.

Laguna XS 2.1 and S 2.1 also support `--draft-model` with their matching DFlash
checkpoint, using 15 proposals by default. See the [Laguna guide](../../../Documentation/Models/Laguna/README.md#draft-models)
for checkpoint pairs and examples.

Sampling and image examples:

```sh
swift run nativ generate --model LiquidAI/LFM2.5-2.6B \
  --draft-model LiquidAI/LFM2.5-2.6B-DSpark --prompt "Write a short story." \
  --temperature 0.7 --top-p 0.9 --seed 42
swift run nativ generate --model mlx-community/Qwen3.5-4B-4bit \
  --draft-model z-lab/Qwen3.5-4B-DFlash --image ./photo.png --prompt "Describe this image."
```

Draft generation defaults to greedy sampling. Video, audio, diffusion, penalties,
and token suppression are unsupported with these draft paths. Use `generate` for
draft requests; interactive `chat` does not accept a draft model. See the
[language guide](../../../Documentation/Models/Text/README.md#speculative-generation)
for checkpoint IDs and SDK session examples.

## Decisions

```sh
swift run nativ predict --model nativ-community/decider-2b \
  --state "Please refund my duplicate charge" --questions-file questions.json
```

Use `--state-file` to read state from a file, or `--json-state` for structured JSON.
Results are written as JSON. [Decider](../../../Documentation/Models/Decider/README.md#cli)
shows the question format; [Laya](../../../Documentation/Models/Laya/README.md)
uses the same format with its own evaluation options.

## Metrics

`generate --json` writes a `BENCH_JSON` metrics record to standard error, including
token counts, timing, and draft acceptance when applicable. Generated output stays
on standard output. For faster repeated use, build once with
`swift build -c release --product nativ` and run `.build/release/nativ`.
