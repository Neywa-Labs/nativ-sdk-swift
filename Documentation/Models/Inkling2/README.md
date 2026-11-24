# Inkling 2 Small

Inkling 2 Small generates text from text and still images. The Thinking Machines
release has 594 billion parameters and uses about 393 GB of checkpoint storage.
Allow additional memory for inference and its KV cache.
[Thinking Machines checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Small)

Use the original checkpoint directory, including `hf_quant_config.json`, or an
Inkling 2 checkpoint converted with mlx-vlm. Generation uses the target model
directly.

## CLI

```sh
swift run nativ generate --model ./Inkling-2-Small \
  --image photo.jpg --prompt "Describe what you see." --max-tokens 256
```

Omit `--image` for text-only prompts. Repeat it to attach more than one image.

## Swift

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(
    from: .directory(URL(fileURLWithPath: "./Inkling-2-Small")))
let input = UserInput(
    chat: [.user(
        "Describe what you see.",
        images: [.url(URL(fileURLWithPath: "photo.jpg"))])],
    additionalContext: ["reasoning_effort": "low"])
let request = model.generate(input: input, options: .init(output: .init(maxTokens: 256)))
print(try await request.response())
```

Set `reasoning_effort` to `none`, `minimal`, `low`, `medium`, `high`, or `xhigh`
in the chat template context. The response stream exposes reasoning, answer text,
and JSON tool calls as separate events.
