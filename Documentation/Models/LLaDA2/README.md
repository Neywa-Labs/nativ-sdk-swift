# LLaDA 2

LLaDA 2 is a diffusion language model family.

- Example checkpoint: `mlx-community/LLaDA2.0-mini-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/LLaDA2.0-mini-4bit \
  --generation-mode diffusion \
  --prompt "The capital of France is" \
  --max-tokens 64
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/LLaDA2.0-mini-4bit")
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

- Use `.diffusion` or `.automatic` generation mode for diffusion decoding.
- This family is text-only in NativSDK.
