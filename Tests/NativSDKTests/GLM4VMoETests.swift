import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct GLM4VMoETestTokenizer: Tokenizer {
    var bosToken: String? {
        nil
    }

    var eosToken: String? {
        "<|endoftext|>"
    }

    var unknownToken: String? {
        nil
    }

    private let tokenIDs: [String: Int] = [
        "<|endoftext|>": 151_329,
        "[gMASK]": 151_331,
        "<sop>": 151_333,
        "<|user|>": 151_336,
        "<|assistant|>": 151_337,
        "<|begin_of_image|>": 151_339,
        "<|end_of_image|>": 151_340,
        "<|image|>": 151_363,
    ]

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        let orderedTokens = tokenIDs.keys.sorted { $0.count > $1.count }
        var output: [Int] = []
        var index = text.startIndex
        while index < text.endIndex {
            if let token = orderedTokens.first(where: { text[index...].hasPrefix($0) }),
                let id = tokenIDs[token]
            {
                output.append(id)
                index = text.index(index, offsetBy: token.count)
            } else {
                output.append(Int(text[index].unicodeScalars.first!.value))
                index = text.index(after: index)
            }
        }
        return output
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
        encode(
            text:
                "[gMASK]<sop>\n<|user|>\n<|begin_of_image|><|image|><|end_of_image|>Describe.\n<|assistant|>\n",
            addSpecialTokens: false,
        )
    }
}

final class GLM4VMoETests: XCTestCase {
    func testProcessorExpandsMoEImagePlaceholder() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            GLM4VProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "Glm46VMoEProcessor",
                  "image_mean": [0.5, 0.5, 0.5],
                  "image_std": [0.5, 0.5, 0.5],
                  "merge_size": 2,
                  "patch_size": 14,
                  "temporal_patch_size": 2,
                  "size": {"shortest_edge": 3136, "longest_edge": 3136}
                }
                """.data(using: .utf8)),
        )
        let processor = GLM4VProcessor(processorConfig, tokenizer: GLM4VMoETestTokenizer())
        let image = CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 56, height: 56))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 151_363 }), 4)
        XCTAssertEqual(tokens.count(where: { $0 == 151_339 }), 1)
        XCTAssertEqual(tokens.count(where: { $0 == 151_340 }), 1)
        XCTAssertEqual(prepared.image?.pixels.shape, [16, 3 * 2 * 14 * 14])
        XCTAssertEqual(prepared.image?.frames?.first?.t, 1)
        XCTAssertEqual(prepared.image?.frames?.first?.h, 4)
        XCTAssertEqual(prepared.image?.frames?.first?.w, 4)
    }

    func testSmallModelInstantiatesAndSanitizeMapsKeys() throws {
        let model = try GLM4VMoE(Self.smallConfig())
        let weights = model.sanitize(weights: [
            "model.visual.downsample.weight": MLXArray.zeros([8, 4, 2, 2]),
            "model.language_model.layers.1.mlp.experts.0.gate_proj.weight": MLXArray.zeros([3, 8]),
            "model.language_model.layers.1.mlp.experts.1.gate_proj.weight": MLXArray.ones([3, 8]),
            "model.language_model.layers.1.mlp.gate.e_score_correction_bias": MLXArray.zeros([2]),
            "model.language_model.layers.1.mlp.switch_mlp.up_proj.weight": MLXArray.zeros([2, 3, 8]
            ),
            "lm_head.weight": MLXArray.zeros([64, 8]),
            "model.language_model.layers.2.self_attn.q_proj.weight": MLXArray.ones([8, 8]),
            "model.language_model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([2]),
        ])

        XCTAssertEqual(weights["vision_tower.downsample.weight"]?.shape, [8, 2, 2, 4])
        XCTAssertEqual(
            weights["language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape,
            [2, 3, 8],
        )
        XCTAssertEqual(
            weights["language_model.model.layers.1.mlp.switch_mlp.up_proj.weight"]?.shape,
            [2, 3, 8],
        )
        XCTAssertEqual(
            weights["language_model.model.layers.1.mlp.gate.e_score_correction_bias"]?.shape,
            [2],
        )
        XCTAssertEqual(weights["language_model.lm_head.weight"]?.shape, [64, 8])
        XCTAssertNil(weights["language_model.model.layers.2.self_attn.q_proj.weight"])
        XCTAssertNil(weights["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
    }

    private static func smallConfig() throws -> GLM4VMoEConfiguration {
        try JSONDecoder.json5().decode(
            GLM4VMoEConfiguration.self,
            from: """
                {
                  "model_type": "glm4v_moe",
                  "image_token_id": 9,
                  "video_token_id": 10,
                  "text_config": {
                    "vocab_size": 64,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "moe_intermediate_size": 3,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "rms_norm_eps": 0.00001,
                    "rope_theta": 10000,
                    "attention_bias": true,
                    "partial_rotary_factor": 0.5,
                    "rope_scaling": {"rope_type": "default", "mrope_section": [1, 0, 0]},
                    "first_k_dense_replace": 1,
                    "n_routed_experts": 2,
                    "n_shared_experts": 1,
                    "num_experts_per_tok": 1,
                    "routed_scaling_factor": 1.0,
                    "norm_topk_prob": true,
                    "n_group": 1,
                    "topk_group": 1,
                    "topk_method": "noaux_tc"
                  },
                  "vision_config": {
                    "hidden_size": 4,
                    "intermediate_size": 10,
                    "depth": 1,
                    "num_heads": 2,
                    "image_size": 4,
                    "patch_size": 2,
                    "out_hidden_size": 8,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "in_channels": 3,
                    "rms_norm_eps": 0.00001
                  }
                }
                """.data(using: .utf8)!,
        )
    }
}
