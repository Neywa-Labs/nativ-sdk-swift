import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct TestTokenizer: Tokenizer {
    let imageTokenID = 151_646

    var bosToken: String? {
        nil
    }

    var eosToken: String? {
        "<|im_end|>"
    }

    var unknownToken: String? {
        nil
    }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        var output: [Int] = []
        var index = text.startIndex
        while index < text.endIndex {
            if text[index...].hasPrefix("<image>") {
                output.append(imageTokenID)
                index = text.index(index, offsetBy: "<image>".count)
            } else {
                output.append(Int(text[index].unicodeScalars.first!.value))
                index = text.index(after: index)
            }
        }
        return output
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIDs.map { id in
            id == imageTokenID ? "<image>" : String(UnicodeScalar(id)!)
        }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        token == "<image>" ? imageTokenID : nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        id == imageTokenID ? "<image>" : nil
    }

    func applyChatTemplate(
        messages _: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?,
    )
        throws -> [Int]
    {
        encode(text: "<image>\nDescribe.", addSpecialTokens: false)
    }
}

final class LLaVATests: XCTestCase {
    func testProcessorExpandsSingleImageTokenForSiglip384() async throws {
        let data = """
            {
              "processor_class": "LlavaProcessor",
              "do_normalize": true,
              "do_rescale": true,
              "do_resize": true,
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5],
              "size": { "height": 384, "width": 384 }
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder.json5().decode(LLaVAProcessorConfiguration.self, from: data)
        let processor = LLaVAProcessor(config, tokenizer: TestTokenizer())
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 32, height: 32))
        let input = UserInput(
            chat: [.user("Describe.", images: [.ciImage(image)])],
        )

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)
        let imageTokens = tokens.filter { $0 == 151_646 }

        XCTAssertEqual(imageTokens.count, 729)
        XCTAssertEqual(prepared.image?.pixels.shape, [1, 3, 384, 384])
    }

    func testSanitizeDropsRotaryAndTransposesPatchEmbedding() throws {
        let data = """
            {
              "model_type": "llava",
              "image_token_index": 9,
              "text_config": {
                "model_type": "qwen2",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 16,
                "tie_word_embeddings": true
              },
              "vision_config": {
                "model_type": "siglip_vision_model",
                "hidden_size": 6,
                "num_hidden_layers": 1,
                "intermediate_size": 12,
                "num_attention_heads": 2,
                "image_size": 4,
                "patch_size": 2,
                "num_channels": 3,
                "layer_norm_eps": 1e-6
              }
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder.json5().decode(LLaVAConfiguration.self, from: data)
        let model = LLaVA(config)

        let weights = model.sanitize(weights: [
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray([1, 2]),
            "vision_tower.vision_model.embeddings.patch_embedding.weight": MLXArray.zeros([
                6, 3, 2, 2,
            ]),
        ])

        XCTAssertNil(weights["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertEqual(
            weights["vision_tower.vision_model.embeddings.patch_embedding.weight"]?.shape,
            [6, 2, 2, 3],
        )
    }
}
