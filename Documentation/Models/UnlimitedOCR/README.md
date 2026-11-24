# Unlimited-OCR and Unlimited-OCR-2

Baidu's Unlimited-OCR models recognize text and layout on document images.
Pass one image for document parsing or pass several page images in reading
order for multi-page parsing. For a PDF, render each page to an image first.

Both official checkpoints load directly. Use
`baidu/Unlimited-OCR` for version 1 and `baidu/Unlimited-OCR-2` for version 2.
For 4-bit weights, use `mlx-community/Unlimited-OCR-4bit` or
`mlx-community/Unlimited-OCR-2-4bit`. Each request processes one document and
uses the default floating-point KV caches.

## CLI

Read a single page with Unlimited-OCR.

```sh
swift run nativ generate --model baidu/Unlimited-OCR \
  --image page-01.png --prompt "document parsing." \
  --temperature 0 --max-tokens 4096
```

Read several pages as one document with Unlimited-OCR-2.

```sh
swift run nativ generate --model baidu/Unlimited-OCR-2 \
  --image page-01.png --image page-02.png \
  --prompt "Multi page parsing." --temperature 0 --max-tokens 8192
```

## Swift

```swift
import Foundation
import NativSDK

let model = try await NativModel.loadModel(from: .repository("baidu/Unlimited-OCR-2"))
let request = model.generate(
    input: UserInput(chat: [
        .user("Multi page parsing.", images: [
            .url(URL(fileURLWithPath: "page-01.png")),
            .url(URL(fileURLWithPath: "page-02.png"))
        ])
    ]),
    options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 8192)))
print(try await request.response())
```

The processor inserts image placeholders and the version 2 page markers.
Both versions use a global view and add local tiles for larger pages. For a
global-view-only request, set `additionalContext: ["cropping": false]` on
`UserInput`.

The response contains the model's layout markup. Unlimited-OCR emits detection
spans and page separators. Unlimited-OCR-2 uses `<PAGE>` and
`<BLOCK>label[x0, y0, x1, y1]<FORK>text`, with continuation labels such as
`cross_text` and `append_text` for paragraphs that span page breaks.

## License

The checkpoints are released under the MIT license. See the
[Unlimited-OCR model card](https://huggingface.co/baidu/Unlimited-OCR),
[Unlimited-OCR-2 model card](https://huggingface.co/baidu/Unlimited-OCR-2),
and the [MLX Python port](https://github.com/Neywa-Labs/mlx-vlm/tree/main/mlx_vlm/models/unlimited_ocr).
