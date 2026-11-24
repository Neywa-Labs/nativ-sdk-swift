# DeepSeek V4 Flash

DeepSeek V4 Flash is a text-only model for chat and text generation.

- Example checkpoint: `mlx-community/DeepSeek-V4-Flash-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/DeepSeek-V4-Flash-4bit \
  --prompt "The capital of France is" \
  --max-tokens 64
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/DeepSeek-V4-Flash-4bit")
let input = UserInput(prompt: .text("The capital of France is"))

try await generateText(source: source, input: input)
```

## Notes

- This is a text-only model family.
- The example uses a DeepSeek V4 Flash checkpoint for text generation.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
