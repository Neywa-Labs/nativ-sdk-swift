# Adapters

Load a compatible LoRA or DoRA adapter to customize a language model's responses.
Start with the base checkpoint the adapter was trained for.

```swift
import NativSDK

let model = try await NativModel.loadModel(
    from: .repository("organization/base-model"))
try await model.loadAdapter(from: .repository("organization/adapter"))
print(try await model.generate(input: UserInput(prompt: "Hello")).response())
try await model.unloadAdapter()
```

Use `.directory(adapterURL)` for local adapters. Repository sources accept a
revision; `loadAdapter` also accepts a custom downloader and a progress callback.
Adapter support depends on the base model.

## Switching and merging

One adapter can be active at a time. Unload it before switching adapters.
`unloadAdapter()` restores the original model. To keep the adapted model without
a separate adapter, call `try await model.fuseAdapter()`.

Fusion changes the loaded model in memory and does not save a checkpoint.
Reload the original model to undo it. Merging into a quantized model can slightly
change its output.

Adapter changes wait for active generation to finish. Existing sessions keep their
conversation history and automatically refresh their caches on the next turn.
Call adapter methods directly on the model, outside `withModel`.

## Supported formats

| Format | Required files |
| --- | --- |
| Native MLX LoRA or DoRA | `adapter_config.json`, `adapters.safetensors` |
| Hugging Face PEFT LoRA | `adapter_config.json`, `adapter_model.safetensors` |

PEFT DoRA, rank-stabilized LoRA, bias updates, per-module rank/scaling overrides,
and additional saved modules are unsupported. Native DoRA supports dense and
quantized models. Adapter training and stacking are not provided.
