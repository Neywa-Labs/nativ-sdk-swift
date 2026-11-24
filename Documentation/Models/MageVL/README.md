# Mage-VL

Mage-VL supports visual question answering and image description. Mage-VL-Cowork
uses the same architecture for screenshot understanding and computer-use prompts.

## Model repositories

| Repository | Notes |
| --- | --- |
| [nativ-community/Mage-VL-OptiQ-4bit](https://huggingface.co/nativ-community/Mage-VL-OptiQ-4bit) | Quantized MLX checkpoint |
| [microsoft/Mage-VL-Cowork](https://huggingface.co/microsoft/Mage-VL-Cowork) | Original BF16 checkpoint; loads directly |

## CLI

```sh
swift run nativ generate \
  --model nativ-community/Mage-VL-OptiQ-4bit \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

## Swift API

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(from: .repository("nativ-community/Mage-VL-OptiQ-4bit"))
let input = UserInput(chat: [.user("Describe this image.", images: [.url(URL(fileURLWithPath: "./image.jpg"))])])
let request = model.generate(
    input: input,
    options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 256)))
for try await event in request {
    if case let .chunk(text) = event {
        print(text, terminator: "")
    }
}
```

## Notes

Text and still images are supported. Video input is unsupported.

## Mage-VL-Cowork

The SDK recognizes the checkpoint's `llava_onevision2` model type and uses the
native Mage-VL model and processor. The original `model.*` language weights and
`visual.*` vision weights load directly without remote Python code.

For computer-use turns, save the upstream
[`MAGE_HYBRID_SYSTEM_PROMPT`](https://huggingface.co/microsoft/Mage-VL-Cowork/blob/main/osworld_eval/mm_agents/mage_hybrid_agent.py)
to `mage_cowork_system.txt` and pass it as a system message:

```swift
let model = try await NativModel.loadModel(
    from: .repository("microsoft/Mage-VL-Cowork"))
let systemPrompt = try String(contentsOfFile: "mage_cowork_system.txt", encoding: .utf8)
let input = UserInput(
    chat: [
        .system(systemPrompt),
        .user("Click the Save button.", images: [.url(screenshotURL)]),
    ],
    additionalContext: ["enable_thinking": true])
let request = model.generate(
    input: input,
    options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 4096))
)
for try await event in request {
    switch event {
    case .chunk(let text): print(text, terminator: "")
    case .toolCall(let call): print(call)
    default: break
    }
}
```

Define `screenshotURL` as the URL of the screenshot to inspect. The system prompt
defines the `computer_use` and `bash` tools and their response format. Coordinates
use the range 0–999. Your application executes requested actions and supplies tool
results and updated screenshots in subsequent chat turns.
The model may request another screenshot before proposing a click.

The original checkpoint contains approximately 9.5 GB of BF16 weights; allow
additional memory for image processing and generation. Model and reference-code
licensing is documented in the [upstream repository](https://huggingface.co/microsoft/Mage-VL-Cowork).
