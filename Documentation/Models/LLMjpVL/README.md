# LLM-jp-VL

LLM-jp-VL supports conversations about images, including prompts in Japanese.

## Model repositories

- [mlx-community/llm-jp-4-vl-9b-mlx-4bit](https://huggingface.co/mlx-community/llm-jp-4-vl-9b-mlx-4bit)

## CLI

```sh
swift run nativ generate \
  --model mlx-community/llm-jp-4-vl-9b-mlx-4bit \
  --image ./image.jpg --prompt "Describe this image." \
  --max-tokens 256
```

## Swift API

```swift
let source = ModelSource.repository("mlx-community/llm-jp-4-vl-9b-mlx-4bit")
let input = UserInput(chat: [
    .user("この画像を説明してください。", images: [.url(URL(fileURLWithPath: "image.jpg"))])
])
try await generateText(source: source, input: input)
```

See the [shared loader example](../README.md#shared-swift-loader) for
`generateText`. Supply the checkpoint's tokenizer and chat template.

Multiple images and conversation history are supported; audio and video are not.
Keep both processor configuration files, the tokenizer, and the chat template
with the checkpoint. Checkpoints using a vision `select_layer` other than `-1`
are unsupported.
