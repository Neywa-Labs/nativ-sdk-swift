# Mistral Small 4

Mistral Small 4 is a text-generation model available through the Mistral 4
family.

- Example checkpoint: `mistralai/Mistral-Small-4-119B-2603`

## CLI

```sh
swift run nativ generate \
  --model mistralai/Mistral-Small-4-119B-2603 \
  --prompt "Question: What is the capital of France? Answer:" \
  --max-tokens 64
```

## Swift

```swift
let source = ModelSource.repository("mistralai/Mistral-Small-4-119B-2603")
let input = UserInput(prompt: .text("Question: What is the capital of France? Answer:"))

try await generateText(source: source, input: input)
```

## Notes

- Select a compatible Mistral text checkpoint with `ModelSource.repository()`.

Use the [shared Swift helper](../README.md#shared-swift-loader) for `generateText`.
