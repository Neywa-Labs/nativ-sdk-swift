# HRM Text

HRM Text is a text-only model family.

- Example checkpoint: `Aryagm/HRM-Text-1B-MLX-4bit`

## CLI

```sh
swift run nativ generate \
  --model Aryagm/HRM-Text-1B-MLX-4bit \
  --prompt "The capital of France is" \
  --max-tokens 64
```

## Swift

```swift
let source = ModelSource.repository("Aryagm/HRM-Text-1B-MLX-4bit")
let input = UserInput(prompt: .text("The capital of France is"))

try await generateText(source: source, input: input)
```

## Notes

- This family is text-only.
- Image, video, and audio inputs are intentionally rejected.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
