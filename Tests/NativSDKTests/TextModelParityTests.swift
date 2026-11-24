import Foundation
import MLX
import XCTest

@testable import NativSDK

final class TextModelParityTests: XCTestCase {
    private static let modelTypes = [
        "mistral",
        "mixtral",
        "llama",
        "phi",
        "phi3",
        "phimoe",
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "gemma4",
        "gemma4_unified",
        "gemma4_text",
        "qwen2",
        "qwen3",
        "qwen3_moe",
        "qwen3_next",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_5_text",
        "nanbeige",
        "minicpm",
        "starcoder2",
        "cohere",
        "openelm",
        "internlm2",
        "deepseek_v2",
        "deepseek_v3",
        "granite",
        "helium",
        "granitemoehybrid",
        "mimo",
        "mimo_v2_flash",
        "minimax",
        "glm4",
        "glm4_moe",
        "glm4_moe_lite",
        "acereason",
        "falcon_h1",
        "bitnet",
        "smollm3",
        "ernie4_5",
        "lfm2",
        "baichuan_m1",
        "exaone4",
        "gpt_oss",
        "lille-130m",
        "olmoe",
        "olmo2",
        "olmo3",
        "bailing_moe",
        "lfm2_moe",
        "nanochat",
        "nemotron_h",
        "afmoe",
        "jamba",
        "mamba2",
        "mistral3",
        "apertus",
        "hunyuan_v1_dense",
    ]

    func testRegisteredTextModelsMatchCachedContinuation() throws {
        for tiedEmbeddings in [false, true] {
            for modelType in Self.modelTypes {
                FileHandle.standardOutput.write(
                    Data(
                        "Checking text model: \(modelType), tied embeddings: \(tiedEmbeddings)\n"
                            .utf8))
                do {
                    MLXRandom.seed(42)
                    let model = try ModelRegistry.default.createModel(
                        configuration: ModelTestConfigurations.textConfiguration(
                            modelType, tiedEmbeddings: tiedEmbeddings), modelType: modelType)
                    try assertTextModelCacheEquivalence(
                        model, tokens: [1, 2, 3, 4], vocabularySize: 96)
                } catch {
                    XCTFail("\(modelType): \(error)")
                }
            }
        }

    }

}
