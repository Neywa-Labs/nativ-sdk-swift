import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct Ernie4_5VLTestTokenizer: Tokenizer {
    private let tokenIDs: [String: Int] = [
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "<|begin_of_sentence|>": 100_273,
        "<|end_of_sentence|>": 100_272,
        "<|IMAGE_PLACEHOLDER|>": 100_295,
        "<|IMAGE_START|>": 101_304,
        "<|IMAGE_END|>": 101_305,
    ]

    var bosToken: String? {
        "<s>"
    }

    var eosToken: String? {
        "</s>"
    }

    var unknownToken: String? {
        "<unk>"
    }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        let orderedTokens = tokenIDs.keys.sorted { $0.count > $1.count }
        var result: [Int] = []
        var index = text.startIndex
        while index < text.endIndex {
            if let token = orderedTokens.first(where: { text[index...].hasPrefix($0) }),
                let id = tokenIDs[token]
            {
                result.append(id)
                index = text.index(index, offsetBy: token.count)
            } else {
                result.append(Int(text[index].unicodeScalars.first!.value))
                index = text.index(after: index)
            }
        }
        return result
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIDs.map { convertIdToToken($0) ?? String(UnicodeScalar($0)!) }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        tokenIDs[token]
    }

    func convertIdToToken(_ id: Int) -> String? {
        tokenIDs.first { $0.value == id }?.key
    }

    func applyChatTemplate(
        messages _: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?,
    )
        throws -> [Int]
    {
        []
    }
}

final class Ernie4_5VLTests: XCTestCase {
    func testProcessorExpandsImagePlaceholdersAndFlattensPatches() async throws {
        let config = try JSONDecoder.json5().decode(
            Ernie4_5VLProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "Ernie4_5_VLProcessor",
                  "spatial_conv_size": 2,
                  "temporal_conv_size": 2,
                  "image_processor": {
                    "min_pixels": 3136,
                    "max_pixels": 3136,
                    "patch_size": 14,
                    "merge_size": 2,
                    "temporal_patch_size": 2,
                    "image_mean": [0.5, 0.5, 0.5],
                    "image_std": [0.5, 0.5, 0.5]
                  }
                }
                """.data(using: .utf8)),
        )
        let processor = Ernie4_5VLProcessor(config, tokenizer: Ernie4_5VLTestTokenizer())
        let image = CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 56, height: 56))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 100_295 }), 4)
        XCTAssertEqual(tokens.count(where: { $0 == 101_304 }), 1)
        XCTAssertEqual(tokens.count(where: { $0 == 101_305 }), 1)
        XCTAssertEqual(prepared.image?.pixels.shape, [16, 3 * 14 * 14])
        XCTAssertEqual(prepared.image?.frames?.first?.t, 1)
        XCTAssertEqual(prepared.image?.frames?.first?.h, 4)
        XCTAssertEqual(prepared.image?.frames?.first?.w, 4)
    }

    func testSmallModelInstantiatesAndSanitizeMapsKeys() throws {
        let model = try Ernie4_5VL(Self.smallConfig())
        let weights = model.sanitize(weights: [
            "vision_model.blocks.0.attn.qkv.weight": MLXArray.zeros([24, 8]),
            "model.resampler_model.spatial_linear.0.weight": MLXArray.zeros([32, 8]),
            "model.layers.0.mlp.experts.0.gate_proj.weight": MLXArray.zeros([4, 8]),
            "model.layers.0.mlp.experts.1.gate_proj.weight": MLXArray.ones([4, 8]),
            "model.layers.0.mlp.experts.2.gate_proj.weight": MLXArray.ones([3, 8]),
            "lm_head.weight": MLXArray.zeros([64, 8]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([2]),
        ])

        XCTAssertEqual(weights["vision_tower.blocks.0.attn.qkv.weight"]?.shape, [24, 8])
        XCTAssertEqual(
            weights["resampler_model.spatial_linear.layers.0.weight"]?.shape,
            [32, 8],
        )
        XCTAssertNil(weights["language_model.lm_head.weight"])
        XCTAssertNil(weights["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNil(weights["language_model.model.layers.0.mlp.experts.0.gate_proj.weight"])
        XCTAssertEqual(
            weights["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape,
            [2, 4, 8],
        )
        XCTAssertEqual(
            weights["language_model.model.layers.0.mlp.switch_mlp_1.gate_proj.weight"]?.shape,
            [1, 3, 8],
        )
    }

    private static func smallConfig() throws -> Ernie4_5VLConfiguration {
        try JSONDecoder.json5().decode(
            Ernie4_5VLConfiguration.self,
            from: """
                {
                  "model_type": "ernie4_5_moe_vl",
                  "hidden_size": 8,
                  "intermediate_size": 16,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 1,
                  "vocab_size": 64,
                  "rms_norm_eps": 0.00001,
                  "rope_theta": 10000,
                  "rope_scaling": {"mrope_section": [1, 1, 0]},
                  "tie_word_embeddings": true,
                  "im_patch_id": 9,
                  "image_start_token_id": 10,
                  "image_end_token_id": 11,
                  "moe_num_experts": [2, 1],
                  "moe_intermediate_size": [4, 3],
                  "moe_layer_start_index": [0, 0],
                  "moe_layer_end_index": [0, 0],
                  "moe_k": 1,
                  "moe_num_shared_experts": 0,
                  "pixel_hidden_size": 8,
                  "spatial_conv_size": 2,
                  "temporal_conv_size": 2,
                  "vision_config": {
                    "depth": 1,
                    "embed_dim": 8,
                    "hidden_size": 8,
                    "mlp_ratio": 2,
                    "num_heads": 2,
                    "in_channels": 3,
                    "patch_size": 2,
                    "spatial_merge_size": 2
                  }
                }
                """.data(using: .utf8)!,
        )
    }
}
