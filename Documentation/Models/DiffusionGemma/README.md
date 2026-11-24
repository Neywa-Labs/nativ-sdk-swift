# Diffusion Gemma

Diffusion Gemma generates text and supports image and video prompts.

- Example checkpoint: `mlx-community/diffusiongemma-26B-A4B-it-4bit`

## CLI

```sh
swift run nativ generate \
  --model mlx-community/diffusiongemma-26B-A4B-it-4bit \
  --generation-mode diffusion \
  --prompt "The capital of France is" \
  --max-tokens 64
```

Vision input:

```sh
swift run nativ generate \
  --model mlx-community/diffusiongemma-26B-A4B-it-4bit \
  --generation-mode diffusion \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 128
```

## Swift

```swift
let source = ModelSource.repository("mlx-community/diffusiongemma-26B-A4B-it-4bit")
let input = UserInput(chat: [
    .user("Describe this image.", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])

let model = try await NativModel.loadModel(
    from: source
)

let request = model.generate(
    input: input,
    options: .init(output: .init(maxTokens: 128), mode: .diffusion)
)
print(try await request.response())
```

## Notes

- Use `.diffusion` or `.automatic` generation mode for diffusion decoding.
- Image and video prompts are supported.
