<h1 align="center">Nativ SDK for Swift</h1>

[![Native SDK for Swift: Run AI models on Apple silicon.](https://raw.githubusercontent.com/Neywa-Labs/mlx-vlm/b7fda105064b609fcc2ef243bd628c3dad317564/docs/assets/mlx-vlm-banner.png)](https://runnativ.ai)

<p align="center">
  <strong>Run AI models on Apple silicon with <a href="https://github.com/ml-explore/mlx-swift">MLX Swift</a>.</strong>
  Generate text and images, create embeddings, and predict decisions on device.
</p>

<p align="center">
  <a href="#installation"><strong>Installation</strong></a> &middot;
  <a href="Documentation/Runtime.md">Usage guide</a> &middot;
  <a href="Documentation/Models/README.md">Supported models</a>
</p>

<p align="center">
  macOS &middot; iOS &middot; iPadOS &middot; tvOS &middot; visionOS
</p>

<p align="center">
  Built by <a href="https://www.neywalabs.ai">Neywa Labs</a>, the team behind <a href="https://runnativ.ai">Nativ</a>.
</p>

<p align="center">
  <a href="Package.swift"><img src="https://img.shields.io/badge/Swift-6.3%2B-orange" alt="Swift 6.3+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license"></a>
</p>

Build on-device chat, visual assistants, search, and image editing into your apps.
Swift concurrency APIs support streaming responses, conversation history, tool calls,
and cancellation. Compatible models accept images, video, and audio alongside text;
see the [model catalog](Documentation/Models/README.md) for supported inputs and checkpoints.

## Installation

Requires **Swift 6.3+** and **macOS 14+, iOS/iPadOS 17+, tvOS 17+, or visionOS 1+**.
Run inference on an Apple silicon Mac or a physical device with a supported Metal GPU.
Choose a model size and quantization that fit your device's available memory.

In Xcode, choose **File → Add Package Dependencies**, enter
`https://github.com/Neywa-Labs/nativ-sdk-swift`, and add the **NativSDK** product to your target.

For a Swift package, add this dependency to `Package.swift`:

```swift
.package(url: "https://github.com/Neywa-Labs/nativ-sdk-swift.git", branch: "main")
```

Then add the product to your target's dependencies:

```swift
.product(name: "NativSDK", package: "nativ-sdk-swift")
```

Models download from Hugging Face on first use and are cached for later requests.
Use `.directory(url)` to load a local checkpoint. Keep the loaded model around to
reuse it across requests; see the [usage guide](Documentation/Runtime.md) for offline
loading, custom download settings, and memory controls.

## Language models

```swift
import NativSDK

let model = try await NativModel.loadModel(
    from: .repository("nativ-community/LFM3-Omni-2B-4bit"))
let response = try await model.generate(
    input: UserInput(prompt: "Why is the sky blue?")
).response()
print(response)
```

To display a response as it arrives:

```swift
for try await event in model.generate(input: UserInput(prompt: "Tell me a short story.")) {
    if case .chunk(let text) = event {
        print(text, terminator: "")
    }
}
```

[Language models: chat, streaming, and generation options](Documentation/Models/Text/README.md)

## Image generation

```swift
import Foundation
import NativSDK

let model = try await NativImageModel.loadModel(
    from: .repository("black-forest-labs/FLUX.2-klein-4B"))
let result = try await model.generate(
    prompt: "A small red apple on a wooden table."
).result()
try result.image.savePNG(to: URL.documentsDirectory.appending(path: "generated.png"))
```

[Image generation: sizes, sampling, and editing](Documentation/Models/ImageGeneration/README.md)

## Embeddings

```swift
import NativSDK

let model = try await NativEmbeddingModel.loadModel(
    from: .repository("sentence-transformers/all-MiniLM-L6-v2"))
let result = try await model.embed(
    input: .text("A small observatory on a mountain.")
).result()
if case .vector(let embedding) = result.embeddings {
    print(embedding)
}
```

[Embeddings: retrieval, input formats, and supported encoders](Documentation/Models/Embeddings/README.md)

## Decisions

```swift
import NativSDK

let model = try await NativDecisionModel.loadModel(
    from: .repository("nativ-community/decider-2b"))
let result = try await model.predict(
    state: "Please refund my duplicate charge",
    questions: ["refund": .bool("Does the customer request a refund?")]
).result()
print(result.answers["refund"]!.value)
```

[Decision models: calibrated answers and prediction options](Documentation/Models/Decisions/README.md)

## Try it out

From a checkout of this repository, start a conversation in the terminal:

```sh
swift run nativ chat --model nativ-community/LFM3-Omni-2B-4bit
```

The [CLI guide](Sources/Tools/nativ/README.md) also covers media prompts, image
generation, and decisions. For a complete SwiftUI app, open the
[Playground](Examples/Playground/README.md) on macOS, iPhone, or iPad to try chat,
image understanding, image generation, and tool calling.

[Usage guide](Documentation/Runtime.md) · [Adapters](Documentation/Adapters.md) · [Supported models](Documentation/Models/README.md)

## Contributing

Bug reports and contributions are welcome on [GitHub](https://github.com/Neywa-Labs/nativ-sdk-swift).
For model issues, include the checkpoint ID, device, and a small reproduction.

Built with [MLX Swift](https://github.com/ml-explore/mlx-swift),
[Swift Transformers](https://github.com/huggingface/swift-transformers), and
[Swift Hugging Face](https://github.com/huggingface/swift-huggingface).

Released under the [MIT license](LICENSE). Model checkpoints have their own licenses.
