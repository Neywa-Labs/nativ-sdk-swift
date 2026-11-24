import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct KimiVLTestTokenizer: Tokenizer {
    private let tokenIDs: [String: Int] = [
        "<|im_system|>": 163_588,
        "<|im_user|>": 163_589,
        "<|im_assistant|>": 163_590,
        "<|im_middle|>": 163_591,
        "<|media_pad|>": 163_592,
        "<|media_start|>": 163_593,
        "<|media_content|>": 163_594,
        "<|media_end|>": 163_595,
        "<|im_end|>": 163_596,
    ]

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
        tokenIDs.map { convertIdToToken($0) ?? "\($0)" }.joined(separator: " ")
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

final class KimiVLTests: XCTestCase {
    func testConfigurationDecodesCheckpointShape() throws {
        let config = try JSONDecoder.json5().decode(
            KimiVLConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "kimi_vl",
                  "media_placeholder_token_id": 163592,
                  "eos_token_id": [163584, 163586],
                  "vocab_size": 163840,
                  "text_config": {
                    "model_type": "deepseek_v3",
                    "vocab_size": 163840,
                    "hidden_size": 2048,
                    "intermediate_size": 11264,
                    "moe_intermediate_size": 1408,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 16,
                    "n_shared_experts": 2,
                    "n_routed_experts": 64,
                    "routed_scaling_factor": 2.446,
                    "kv_lora_rank": 512,
                    "q_lora_rank": null,
                    "qk_rope_head_dim": 64,
                    "v_head_dim": 128,
                    "qk_nope_head_dim": 128,
                    "n_group": 1,
                    "topk_group": 1,
                    "num_experts_per_tok": 6,
                    "first_k_dense_replace": 1,
                    "rms_norm_eps": 0.00001,
                    "rope_theta": 800000
                  },
                  "vision_config": {
                    "model_type": "moonvit",
                    "patch_size": 14,
                    "init_pos_emb_height": 64,
                    "init_pos_emb_width": 64,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "merge_kernel_size": [2, 2]
                  }
                }
                """.data(using: .utf8)),
        )

        XCTAssertEqual(config.modelType, "kimi_vl")
        XCTAssertEqual(config.mediaPlaceholderTokenID, 163_592)
        XCTAssertEqual(config.imageTokenIndex, 163_592)
        XCTAssertEqual(config.eosTokenIDs, [163_584, 163_586])
        XCTAssertEqual(config.textConfiguration.modelType, "deepseek_v3")
        XCTAssertEqual(config.textConfiguration.hiddenSize, 2048)
        XCTAssertEqual(config.textConfiguration.kvLoraRank, 512)
        XCTAssertNil(config.textConfiguration.qLoraRank)
        XCTAssertEqual(config.textConfiguration.numExpertsPerToken, 6)
        XCTAssertEqual(config.visionConfiguration.modelType, "moonvit")
        XCTAssertEqual(config.visionConfiguration.hiddenLayers, 27)
        XCTAssertEqual(config.visionConfiguration.attentionHeads, 16)
        XCTAssertEqual(config.visionConfiguration.mergeKernelSize, [2, 2])
    }

    func testProcessorPacksPatchesAndExpandsMediaPadTokens() async throws {
        let config = try JSONDecoder.json5().decode(
            KimiVLProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "KimiVLProcessor",
                  "patch_size": 14,
                  "pad_input": true,
                  "in_token_limit": 4096,
                  "merge_kernel_size": [2, 2],
                  "image_mean": [0.5, 0.5, 0.5],
                  "image_std": [0.5, 0.5, 0.5]
                }
                """.data(using: .utf8)),
        )
        let processor = KimiVLProcessor(config, tokenizer: KimiVLTestTokenizer())
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 12, height: 10))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 163_592 }), 1)
        XCTAssertTrue(tokens.contains(163_593))
        XCTAssertTrue(tokens.contains(163_594))
        XCTAssertTrue(tokens.contains(163_595))
        XCTAssertEqual(prepared.image?.pixels.shape, [4, 3, 14, 14])
        XCTAssertEqual(prepared.image?.frames?.first?.t, 1)
        XCTAssertEqual(prepared.image?.frames?.first?.h, 2)
        XCTAssertEqual(prepared.image?.frames?.first?.w, 2)
    }

    func testSmallModelInstantiatesAndSanitizeMapsKeys() throws {
        let model = try KimiVL(Self.smallConfig())
        let weights = model.sanitize(weights: [
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([2]),
            "vision_tower.encoder.blocks.0.wqkv.weight": MLXArray.zeros([24, 8]),
            "vision_tower.encoder.blocks.0.wo.bias": MLXArray.zeros([8]),
            "vision_tower.patch_embed.proj.weight": MLXArray.zeros([8, 3, 2, 2]),
            "language_model.model.layers.1.mlp.experts.0.gate_proj.weight": MLXArray.zeros([4, 8]),
            "language_model.model.layers.1.mlp.experts.1.gate_proj.weight": MLXArray.zeros([4, 8]),
        ])

        XCTAssertNil(weights["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertEqual(weights["vision_tower.blocks.0.attn.wqkv.weight"]?.shape, [24, 8])
        XCTAssertEqual(weights["vision_tower.blocks.0.attn.wo.bias"]?.shape, [8])
        XCTAssertEqual(weights["vision_tower.patch_embed.proj.weight"]?.shape, [8, 2, 2, 3])
        XCTAssertNil(weights["language_model.model.layers.1.mlp.experts.0.gate_proj.weight"])
        XCTAssertEqual(
            weights["language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape,
            [2, 4, 8],
        )
    }

    private static func smallConfig() throws -> KimiVLConfiguration {
        try JSONDecoder.json5().decode(
            KimiVLConfiguration.self,
            from: """
                {
                  "model_type": "kimi_vl",
                  "media_placeholder_token_id": 31,
                  "vocab_size": 64,
                  "text_config": {
                    "vocab_size": 64,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "moe_intermediate_size": 4,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "n_shared_experts": null,
                    "n_routed_experts": 2,
                    "routed_scaling_factor": 1,
                    "kv_lora_rank": 4,
                    "q_lora_rank": null,
                    "qk_rope_head_dim": 2,
                    "v_head_dim": 2,
                    "qk_nope_head_dim": 2,
                    "n_group": 1,
                    "topk_group": 1,
                    "num_experts_per_tok": 1,
                    "first_k_dense_replace": 1,
                    "rms_norm_eps": 0.00001,
                    "rope_theta": 10000,
                    "max_position_embeddings": 16
                  },
                  "vision_config": {
                    "model_type": "moonvit",
                    "hidden_size": 8,
                    "embed_dim": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "patch_size": 2,
                    "num_channels": 3,
                    "init_pos_emb_height": 2,
                    "init_pos_emb_width": 2,
                    "merge_kernel_size": [2, 2]
                  }
                }
                """.data(using: .utf8)!,
        )
    }
}
