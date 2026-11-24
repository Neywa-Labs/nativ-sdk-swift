# Nativ SDK Playground

A macOS, iPhone, and iPad playground for exploring and testing NativSDK features,
including text conversations, image understanding, image generation, and tool
calling. It uses the local NativSDK package with
[LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B),
[LiquidAI/LFM2.5-VL-3B](https://huggingface.co/LiquidAI/LFM2.5-VL-3B), and
[Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1).

## Run

Requires Xcode with Swift 6.3+ and
[XcodeGen](https://github.com/yonaskolb/XcodeGen).

From this directory:

```sh
xcodegen generate
open NativSDKPlayground.xcodeproj
```

- **macOS 14+ (Apple silicon):** Select **NativSDKPlayground** and **My Mac**,
  then run. No development team is required.
- **iOS/iPadOS 17+:** Select **NativSDKPlayground-iOS** and your iPhone or iPad.
  Choose your development team under the iOS target’s **Signing & Capabilities**,
  then run. Chat, Vision, Images, settings, and PNG export are available on both platforms.
- **Simulator:** Use **NativSDKPlayground-iOS** to inspect the interface. Model
  loading and generation show a physical-device notice without downloading weights.
  [MLX inference requires a physical device’s Metal GPU](https://github.com/ml-explore/mlx-swift/blob/0.33.0/Source/MLX/Documentation.docc/Articles/running-on-ios.md).

Regenerate the project after changing `project.yml` or adding source files.

The first launch downloads the text model (roughly 5.4 GB). Vision downloads its
own model on first use, and image generation downloads its model on the first
request. Later launches reuse downloaded files. Allow additional unified memory
for generation and conversation history; Qwen-Image requires substantially more
memory and disk space than the Liquid models. On iOS, available app memory depends
on the device; these checkpoints may exceed it, especially Qwen-Image. The iOS
target requests the Increased Memory Limit entitlement for supported devices,
but this does not guarantee that every model will fit.

Messages are processed locally and are not saved between app launches. The menu
keyboard shortcuts below apply to macOS; use the onscreen controls on iOS.

## Chat

Type a message and tap or click **Send**, or press **⌘ Return** with a keyboard.
Replies stream as they are generated. **Stop** (Escape) cancels the turn and restores its prompt for
editing. **New Chat** (⌘ N) clears the conversation while keeping the model loaded.

Open **Chat Settings** in the toolbar (or press **⌘ ,**) and enable **Use Draft Model** to use the matching
[text DSpark](https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark) or
[vision DSpark](https://huggingface.co/LiquidAI/LFM2.5-VL-3B-DSpark) model.
The additional download is roughly 655 MB for Text or 559 MB for Vision.
Changing this setting starts a new conversation. It remains enabled when switching
between Chat and Vision. Drafting works with streamed replies, image prompts,
and the weather tool. Expand **Response details** to see speed and accepted/proposed draft tokens.

## Try a tool

In **Chat** mode, select **Try the weather tool** or ask “What’s the weather
in Paris?” Tool arguments and results appear in the conversation before the model's
follow-up answer. Expand **Simulated weather** to see tool arguments and results.
This demo returns simulated weather—21°C and sunny—for every
city and makes no network requests. Stopping or failing a turn restores the
conversation to before that prompt.

## Describe an image

Select **Vision**, then select **Describe the sample image**. The bundled image
appears with your prompt, and you can ask follow-up questions about it. Stopping
restores both the prompt and image for resending.

## Generate an image

Select **Images**, describe an image, and select **Generate** or press
**⌘ Return**. Open **Image Options** below the prompt (or press **⌘ ,**) to adjust
size, steps, and seed. Requests are independent of
the chat conversation. Loading shows download progress; generation shows elapsed
time. **Stop** cancels the request and keeps your prompt and previous image.
Use **Save PNG…** in the toolbar, or **⌘ S**, to export a completed image.

Switching modes releases the previous model. Leaving chat clears its conversation;
leaving image generation keeps the prompt and preview.

## Use these features in your app

See the [language guide](../../Documentation/Models/Text/README.md) for streaming,
sessions, and draft models, the [runtime guide](../../Documentation/Runtime.md#media-and-tools)
for media and tools, and the [image guide](../../Documentation/Models/ImageGeneration/README.md)
for image generation and cancellation.
