# Nemotron Labs Diffusion

Nemotron Labs Diffusion is a text-only generation model.

- Example checkpoint: `mlx-community/Nemotron-Labs-Diffusion-3B-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/Nemotron-Labs-Diffusion-3B-4bit \
  --generation-mode diffusion \
  --prompt "The capital of France is" \
  --max-tokens 64
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/Nemotron-Labs-Diffusion-3B-4bit")
let input = UserInput(prompt: .text("The capital of France is"))

let model = try await NativModel.loadModel(
    from: source
)

let request = model.generate(
    input: input,
    options: .init(output: .init(maxTokens: 64), mode: .diffusion)
)
print(try await request.response())
```

## Notes

- Use diffusion generation mode for this family.
- This family is text-only.
