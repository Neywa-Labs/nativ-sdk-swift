# Supported models

Find checkpoint IDs, Swift examples, CLI commands, and input requirements for
Nativ SDK models.

## Model kinds

- [Language models](Text/README.md): chat, streaming, and draft models.
- [Image generation](ImageGeneration/README.md): image creation and editing.
- [Embeddings](Embeddings/README.md): search and retrieval vectors.
- [Decision models](Decisions/README.md): named choice, Boolean, and score answers.
- [OCR scoring](MiniCPMOCRJev/README.md): verify candidate OCR against document crops.

## Shared Swift Loader

The language-model examples that call `generateText` use this helper:

```swift
import Foundation
import NativSDK

func generateText(source: ModelSource, input: sending UserInput) async throws {
    let model = try await NativModel.loadModel(from: source)
    print(try await model.generate(input: input).response())
}
```

Use `.repository("org/repo")` for a Hugging Face checkpoint or `.directory(url)`
for local files. Follow the family guide's loading instructions when a model needs
a nested checkpoint directory or additional metadata.
Choose a model size and quantization that fit your available memory.

## Catalog

| Family | README |
| --- | --- |
| Aya Vision | [AyaVision](AyaVision/README.md) |
| BERT / MiniLM embeddings | [BERT](BERT/README.md) |
| Bunny Llama | [BunnyLlama](BunnyLlama/README.md) |
| Cohere2 MoE / North Mini Code | [Cohere2MoE](Cohere2MoE/README.md) |
| Decider | [Decider](Decider/README.md) |
| DeepSeek OCR | [DeepSeekOCR](DeepSeekOCR/README.md) |
| DeepSeek OCR 2 | [DeepSeekOCR2](DeepSeekOCR2/README.md) |
| DeepSeek V4 Flash | [DeepSeekV4](DeepSeekV4/README.md) |
| DeepSeek VL | [DeepSeekVL](DeepSeekVL/README.md) |
| DeepSeek VL2 | [DeepSeekVL2](DeepSeekVL2/README.md) |
| Diffusion Gemma | [DiffusionGemma](DiffusionGemma/README.md) |
| dots.ocr | [DotsOCR](DotsOCR/README.md) |
| EmbeddingGemma 2 | [EmbeddingGemma2](EmbeddingGemma2/README.md) |
| ERNIE-Image Base and Turbo | [ErnieImage](ErnieImage/README.md) |
| ERNIE 4.5 VL | [Ernie45VL](Ernie45VL/README.md) |
| ERNIE 5.5 Flash Base | [Ernie55](Ernie55/README.md) |
| Falcon OCR | [FalconOCR](FalconOCR/README.md) |
| Falcon Perception | [FalconPerception](FalconPerception/README.md) |
| Flint | [Flint](Flint/README.md) |
| Florence 2 | [Florence2](Florence2/README.md) |
| FLUX.2 | [Flux2](Flux2/README.md) |
| FLUX.3 Image Klein | [Flux3](Flux3/README.md) |
| Gemma 3n | [Gemma3n](Gemma3n/README.md) |
| Gemma 4 Unified | [Gemma4Unified](Gemma4Unified/README.md) |
| GLM 4.1V | [GLM41V](GLM41V/README.md) |
| GLM 4.5V | [GLM45V](GLM45V/README.md) |
| GLM-5-Next | [GLM5Next](GLM5Next/README.md) |
| GLM-OCR | [GLMOCR](GLMOCR/README.md) |
| Granite 4 Vision | [Granite4Vision](Granite4Vision/README.md) |
| Granite Vision 3.2 | [GraniteVision](GraniteVision/README.md) |
| HRM Text | [HRMText](HRMText/README.md) |
| Idefics2 | [Idefics2](Idefics2/README.md) |
| Ideogram 4 | [Ideogram4](Ideogram4/README.md) |
| IndicOCR (OCR stage) | [IndicOCR](IndicOCR/README.md) |
| InternVL3 | [InternVL3](InternVL3/README.md) |
| Jina VLM | [JinaVLM](JinaVLM/README.md) |
| K2-Horizon | [K2Horizon](K2Horizon/README.md) |
| Kimi VL | [KimiVL](KimiVL/README.md) |
| Laguna | [Laguna](Laguna/README.md) |
| Laya | [Laya](Laya/README.md) |
| LFM3-Audio / LFM3-Omni | [LFM3](LFM3/README.md) |
| Ling 3.1 Flash | [Ling31Flash](Ling31Flash/README.md) |
| LLaDA 2 | [LLaDA2](LLaDA2/README.md) |
| Inkling 2 Small | [Inkling2](Inkling2/README.md) |
| LLaDA-Image | [LLaDAImage](LLaDAImage/README.md) |
| Llama 3.2 Vision | [Llama32Vision](Llama32Vision/README.md) |
| Llama 4 Scout | [Llama4Scout](Llama4Scout/README.md) |
| LLaVA Interleave | [LLaVAInterleave](LLaVAInterleave/README.md) |
| LLaVA Next | [LLaVANext](LLaVANext/README.md) |
| LLaVA-OneVision | [LLaVAOneVision](LLaVAOneVision/README.md) |
| LLM-jp-VL | [LLMjpVL](LLMjpVL/README.md) |
| Locate Anything | [LocateAnything](LocateAnything/README.md) |
| LongCat-Flash-Lite-Sparse | [LongCatFlashSparse](LongCatFlashSparse/README.md) |
| Mage-VL / Mage-VL-Cowork | [MageVL](MageVL/README.md) |
| Mage-Flow | [MageFlow](MageFlow/README.md) |
| Ming-Image 0.1 Design | [MingImage](MingImage/README.md) |
| MiniCPM-o | [MiniCPMO](MiniCPMO/README.md) |
| MiniCPM-OCR 1.0 | [MiniCPMOCR](MiniCPMOCR/README.md) |
| MiniCPM-OCR Jev | [MiniCPMOCRJev](MiniCPMOCRJev/README.md) |
| OXR-1.0 | [OXR](OXR/README.md) |
| Nemotron 3.5 Parse | [NemotronParse35](NemotronParse35/README.md) |
| MiniCPM-V | [MiniCPMV](MiniCPMV/README.md) |
| MiniMax-H3-Image | [MiniMaxH3Image](MiniMaxH3Image/README.md) |
| MiniMax M3 | [MiniMaxM3](MiniMaxM3/README.md) |
| Mistral 3 | [Mistral3](Mistral3/README.md) |
| Mistral Large 3 | [MistralLarge3](MistralLarge3/README.md) |
| Mistral Large 4 | [MistralLarge4](MistralLarge4/README.md) |
| Mistral Small 4 | [Mistral4](Mistral4/README.md) |
| Molmo | [Molmo](Molmo/README.md) |
| Molmo2 | [Molmo2](Molmo2/README.md) |
| MolmoPoint | [MolmoPoint](MolmoPoint/README.md) |
| Moondream2 | [Moondream2](Moondream2/README.md) |
| Moondream3 | [Moondream3](Moondream3/README.md) |
| MOSS-Image Base and SFT | [MOSSImage](MOSSImage/README.md) |
| Muse Spark Open | [MuseSpark](MuseSpark/README.md) |
| Nemotron Labs Diffusion | [NemotronLabsDiffusion](NemotronLabsDiffusion/README.md) |
| Nemotron Omni | [NemotronOmni](NemotronOmni/README.md) |
| Nemotron 3.5 Transcribe | [NemotronTranscribe](NemotronTranscribe/README.md) |
| Nemotron-Parse | [NemotronParse](NemotronParse/README.md) |
| PaddleOCR-VL | [PaddleOCRVL](PaddleOCRVL/README.md) |
| Phi-3.5 Vision | [Phi35Vision](Phi35Vision/README.md) |
| Phi-4 Multimodal | [Phi4Multimodal](Phi4Multimodal/README.md) |
| Phi-4 Reasoning Vision | [Phi4SigLIP](Phi4SigLIP/README.md) |
| Prism Bonsai | [Bonsai](Bonsai/README.md) |
| Qwen-Image 2.1 | [QwenImage](QwenImage/README.md) |
| Qwen3 Omni | [Qwen3Omni](Qwen3Omni/README.md) |
| Qwen3.5 Vision-Language | [Qwen35](Qwen35/README.md) |
| Qwen3.8-Flash-Next | [Qwen4Exp](Qwen4Exp/README.md) |
| Raon-1 Mini | [Raon1Mini](Raon1Mini/README.md) |
| Ripple-65B-Preview | [Ripple](Ripple/README.md) |
| Spark-X2.5 | [Spark25](Spark25/README.md) |
| Step-3.7 Flash | [Step37Flash](Step37Flash/README.md) |
| Ternary Bonsai 2 | [TernaryBonsai2](TernaryBonsai2/README.md) |
| Text models | [Text](Text/README.md) |
| Unlimited-OCR and Unlimited-OCR-2 | [UnlimitedOCR](UnlimitedOCR/README.md) |
| Youtu VL | [YoutuVL](YoutuVL/README.md) |
| Z-Image | [ZImage](ZImage/README.md) |
| Z1T-0 | [Z1T](Z1T/README.md) |
| ZAYA1-VL | [Zaya1VL](Zaya1VL/README.md) |

## Additional multimodal models

These families use `NativModel.loadModel` and the response APIs above:

| Family | Supported inputs |
| --- | --- |
| Cohere Compass | Text and still images |
| DeepSeek V4.1 | Text and still images |
| Hunyuan VL | Text and still images |
| InternVL | Text and still images |
| Kimi K3 | Text and still images |
| MiMo V2 | Text and still images |

Still-image support does not imply video or audio support. See individual family
guides for supported media, draft models, and any checkpoint restrictions.
