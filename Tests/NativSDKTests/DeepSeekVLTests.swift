import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct DeepSeekVLTestTokenizer: Tokenizer {
    let imageTokenID = 100_015

    var bosToken: String? {
        "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>"
    }

    var eosToken: String? {
        "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>"
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
        switch token {
        case "<image>": imageTokenID
        case bosToken: 100_000
        case eosToken: 100_001
        default: nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch id {
        case imageTokenID: "<image>"
        case 100_000: bosToken
        case 100_001: eosToken
        default: nil
        }
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

final class DeepSeekVLTests: XCTestCase {
    func testConfigurationDecodesDeepSeekVLShape() throws {
        let data = """
            {
              "aligner_config": {
                "cls": "MlpProjector",
                "model_type": "aligner",
                "params": {
                  "depth": 2,
                  "input_dim": 1024,
                  "n_embed": 2048,
                  "projector_type": "mlp_gelu"
                }
              },
              "text_config": {
                "hidden_size": 2048,
                "intermediate_size": 5632,
                "max_position_embeddings": 16384,
                "model_type": "llama",
                "num_attention_heads": 16,
                "num_hidden_layers": 24,
                "num_key_value_heads": 16,
                "torch_dtype": "float16",
                "vocab_size": 102400
              },
              "model_type": "multi_modality",
              "vision_config": {
                "cls": "CLIPVisionTower",
                "model_type": "vision",
                "params": {
                  "image_size": 384,
                  "model_name": "siglip_large_patch16_384",
                  "pixel_mean": [0.5, 0.5, 0.5],
                  "pixel_std": [0.5, 0.5, 0.5],
                  "select_feature": "same",
                  "select_layer": -1
                },
                "intermediate_size": 4096
              }
            }
            """.data(using: .utf8)!

        let config = try JSONDecoder.json5().decode(DeepSeekVLConfiguration.self, from: data)

        XCTAssertEqual(config.modelType, "multi_modality")
        XCTAssertEqual(config.textConfiguration.modelType, "llama")
        XCTAssertEqual(config.textConfiguration.hiddenSize, 2048)
        XCTAssertEqual(config.textConfiguration.rmsNormEps, 1e-6)
        XCTAssertEqual(config.visionConfiguration.cls, "CLIPVisionTower")
        XCTAssertEqual(config.visionConfiguration.hiddenSize, 1024)
        XCTAssertEqual(config.visionConfiguration.imageSize, 384)
        XCTAssertEqual(config.imageTokenIndex, 100_015)
        XCTAssertEqual(config.numImageTokens, 576)
    }

    func testProcessorExpandsImagePromptAndPadsToSquare() async throws {
        let data = """
            {
              "background_color": [127, 127, 127],
              "do_normalize": true,
              "image_mean": [0.5, 0.5, 0.5],
              "image_processor_type": "VLMImageProcessor",
              "image_size": 384,
              "image_std": [0.5, 0.5, 0.5],
              "min_size": 14,
              "processor_class": "VLChatProcessor",
              "rescale_factor": 0.00392156862745098
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder.json5().decode(
            DeepSeekVLProcessorConfiguration.self, from: data)
        let processor = DeepSeekVLProcessor(config, tokenizer: DeepSeekVLTestTokenizer())
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 640, height: 320))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 100_015 }), 576)
        XCTAssertEqual(prepared.image?.pixels.shape, [1, 3, 384, 384])
    }

    func testSanitizeDropsRotaryAndTransposesPatchEmbedding() throws {
        let data = """
            {
              "aligner_config": {
                "cls": "MlpProjector",
                "model_type": "aligner",
                "params": { "depth": 2, "projector_type": "mlp_gelu" }
              },
              "text_config": {
                "model_type": "llama",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 16
              },
              "model_type": "multi_modality",
              "vision_config": {
                "model_type": "vision",
                "hidden_size": 6,
                "num_hidden_layers": 1,
                "intermediate_size": 12,
                "num_attention_heads": 2,
                "image_size": 4,
                "patch_size": 2,
                "num_channels": 3,
                "layer_norm_eps": 1e-6,
                "cls": "CLIPVisionTower",
                "params": { "image_size": 4 }
              }
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder.json5().decode(DeepSeekVLConfiguration.self, from: data)
        let model = DeepSeekVL(config)

        let weights = model.sanitize(weights: [
            "aligner.layers.0.weight": MLXArray.zeros([8, 6]),
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray([1, 2]),
            "vision_model.vision_tower.patch_embed.proj.weight": MLXArray.zeros([
                6, 3, 2, 2,
            ]),
        ])

        XCTAssertNil(weights["aligner.layers.0.weight"])
        XCTAssertEqual(weights["aligner.linear0.weight"]?.shape, [8, 6])
        XCTAssertNil(weights["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertEqual(
            weights["vision_model.vision_tower.patch_embed.proj.weight"]?.shape,
            [6, 2, 2, 3],
        )
    }
}
