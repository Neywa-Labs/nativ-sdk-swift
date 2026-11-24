# Laguna

Laguna is a text-only model family.

- Example checkpoint: `mlx-community/Laguna-XS.2-8bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/Laguna-XS.2-8bit \
  --prompt "Write a short Swift function that adds two integers." \
  --max-tokens 160
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/Laguna-XS.2-8bit")
let input = UserInput(prompt: .text("Write a short Swift function that adds two integers."))

try await generateText(source: source, input: input)
```

## Notes

- Laguna is text-only.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.

## Draft models

Laguna XS 2.1 and S 2.1 support DFlash generation with their matching draft:

| Target | Draft |
| --- | --- |
| Laguna XS 2.1, including MLX quantizations | `poolside/Laguna-XS-2.1-DFlash` |
| Laguna S 2.1, including MLX quantizations | `poolside/Laguna-S-2.1-DFlash` |

```swift
let model = try await NativDraftModel.loadModel(
    target: .repository("mlx-community/Laguna-XS-2.1-4bit"),
    draft: .repository("poolside/Laguna-XS-2.1-DFlash"))
print(try await model.generate(
    input: UserInput(prompt: "Write a Swift function that adds two integers.")
).response())
```

For a conversation, pass the loaded pair to `ModelSession(model: model)`.

```sh
swift run nativ generate \
  --model mlx-community/Laguna-XS-2.1-4bit \
  --draft-model poolside/Laguna-XS-2.1-DFlash \
  --prompt "Write a Swift function that adds two integers."
```

Both drafts default to 15 proposals per round. Set `--draft-tokens` or
`speculation: .init(draftTokenCount: 4)` to use fewer. Greedy and temperature
sampling are supported; responses may differ from ordinary generation even with
the same seed. See [draft generation options](../Text/README.md#speculative-generation)
for sampling settings and supported cache options.
