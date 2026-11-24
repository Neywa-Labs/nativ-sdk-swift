# ERNIE 5.5 Flash Base

ERNIE 5.5 Flash Base is a text completion model. Start with the text you want
the model to continue. The original FP8 checkpoint uses about 288 GB of storage.
Allow additional memory for loading and for the attention caches.
[Baidu model card](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base)

Use a Baidu FP8 or BF16 checkpoint directory, or an mlx-vlm-converted checkpoint.
The loader keeps supported FP8 projections in MLX's MXFP8 layout.

## CLI

```sh
swift run nativ generate --model ./ERNIE-5.5-Flash-Base \
  --prompt "Large language models are" --temperature 0 --max-tokens 128
```

## Swift

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(
    from: .directory(URL(fileURLWithPath: "./ERNIE-5.5-Flash-Base")))
let request = model.generate(
    input: UserInput(prompt: "Large language models are"),
    options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 128)))
print(try await request.response())
```

The SDK uses the target model for autoregressive decoding. Use the default
cache configuration for the compressed attention and latent KV state.

## License

Baidu distributes the weights under the ERNIE 5.5 Flash Base Model License.
[Model license](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base/blob/main/LICENSE)

The Python reference is part of the MIT-licensed mlx-vlm project.
[Python implementation](https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/ernie5_5_moe) ·
[MIT license](https://github.com/Blaizzy/mlx-vlm/blob/main/LICENSE)
