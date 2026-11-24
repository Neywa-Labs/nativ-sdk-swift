# Cohere2 MoE / North Mini Code

North Mini Code is a text-only model for coding tasks.

- Example checkpoint: `mlx-community/North-Mini-Code-1.0-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/North-Mini-Code-1.0-4bit \
  --prompt "Write a short Swift function that adds two integers." \
  --max-tokens 160
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/North-Mini-Code-1.0-4bit")
let input = UserInput(prompt: .text("Write a short Swift function that adds two integers."))

try await generateText(source: source, input: input)
```

## Notes

- Text-only models reject image, video, and audio inputs.
- Use a plain text prompt or chat-style `UserInput` without media attachments.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
