import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct Zaya1VLTestTokenizer: Tokenizer {
    private let tokenIDs: [String: Int] = [
        "<pad>": 0,
        "<|im_start|>": 90,
        "<|im_end|>": 91,
        "<|vision_start|>": 92,
        "<|vision_end|>": 93,
        "<image>": 99,
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
        let specials = tokenIDs.keys.sorted { $0.count > $1.count }
        var result: [Int] = []
        var index = text.startIndex
        while index < text.endIndex {
            if let token = specials.first(where: { text[index...].hasPrefix($0) }),
                let id = tokenIDs[token]
            {
                result.append(id)
                index = text.index(index, offsetBy: token.count)
            } else {
                result.append(Int(text[index].unicodeScalars.first!.value) % 64)
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

final class Zaya1VLTests: XCTestCase {
    func testConfigurationDecodesCheckpointShape() throws {
        let config = try JSONDecoder.json5().decode(
            Zaya1VLConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "zaya1_vl",
                  "activation_func": "swiglu",
                  "attention_bias": false,
                  "bos_token_id": 2,
                  "cca": true,
                  "cca_time0": 2,
                  "cca_time1": 2,
                  "eos_token_id": 262143,
                  "ffn_hidden_size": 4096,
                  "hidden_size": 2048,
                  "image_token_id": 262147,
                  "max_position_embeddings": 32768,
                  "moe_router_topk": 1,
                  "norm_epsilon": 0.00001,
                  "num_attention_heads": 8,
                  "num_experts": 16,
                  "num_hidden_layers": 40,
                  "num_key_value_heads": 2,
                  "num_query_groups": 2,
                  "pad_token_id": 0,
                  "rope_pct": 0.5,
                  "rotary_base": 1000000,
                  "scale_residual_merge": true,
                  "tie_word_embeddings": true,
                  "vision_lora": true,
                  "vision_lora_rank_attn": 8,
                  "vision_lora_rank_mlp": 32,
                  "vocab_size": 262272,
                  "zaya_mlp_expansion": 256,
                  "zaya_use_eda": true,
                  "zaya_use_mod": true,
                  "vision_config": {
                    "hidden_size": 1280,
                    "in_chans": 3,
                    "model_type": "qwen2_5_vl",
                    "out_hidden_size": 2048,
                    "spatial_patch_size": 14,
                    "temporal_patch_size": 1,
                    "tokens_per_second": 2
                  }
                }
                """.data(using: .utf8)),
        )

        XCTAssertEqual(config.modelType, "zaya1_vl")
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 40)
        XCTAssertEqual(config.textConfiguration.kvHeads, 2)
        XCTAssertEqual(config.textConfiguration.ffnHiddenSizeOut, 2048)
        XCTAssertEqual(config.imageTokenID, 262_147)
        XCTAssertEqual(config.visionStartTokenID, 255_999)
        XCTAssertEqual(config.visionConfiguration.depth, 32)
        XCTAssertEqual(config.visionConfiguration.hiddenSize, 1280)
        XCTAssertEqual(config.visionConfiguration.outHiddenSize, 2048)
        XCTAssertEqual(config.visionConfiguration.patchSize, 14)
        XCTAssertEqual(config.visionConfiguration.temporalPatchSize, 1)
    }

    func testProcessorPacksQwenStylePatchesAndExpandsImageTokens() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            Zaya1VLProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "Zaya1VLProcessor",
                  "image_mean": [0.5, 0.5, 0.5],
                  "image_std": [0.5, 0.5, 0.5],
                  "min_pixels": 784,
                  "max_pixels": 784,
                  "merge_size": 2,
                  "patch_size": 14,
                  "temporal_patch_size": 1
                }
                """.data(using: .utf8)),
        )
        let processor = Zaya1VLProcessor(processorConfig, tokenizer: Zaya1VLTestTokenizer())
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 28, height: 28))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 99 }), 1)
        XCTAssertTrue(tokens.contains(92))
        XCTAssertTrue(tokens.contains(93))
        XCTAssertEqual(prepared.image?.pixels.shape, [4, 588])
        XCTAssertEqual(prepared.image?.frames?.first?.t, 1)
        XCTAssertEqual(prepared.image?.frames?.first?.h, 2)
        XCTAssertEqual(prepared.image?.frames?.first?.w, 2)
    }

    func testSmallModelRunsVisionToLanguagePrepareWithCacheList() throws {
        let config = try Self.smallConfig()
        let model = Zaya1VL(config)
        let tokens = MLXArray([90, 92, 99, 93, 91]).reshaped(1, 5)
        let input = LMInput(
            text: .init(tokens: tokens, mask: ones(like: tokens).asType(.int8)),
            image: .init(pixels: MLXArray.zeros([4, 588]), frames: [THW(1, 2, 2)]),
        )
        let cache = try model.newCache(configuration: nil)

        XCTAssertTrue(cache.first is CacheList)

        let prepared = try model.prepare(input, cache: cache, state: nil, prefill: .init())
        switch prepared {
        case .logits(let output):
            XCTAssertEqual(output.logits.shape, [1, 5, 128])
        case .tokens:
            XCTFail("Zaya1VL.prepare should return logits")
        }
    }

    func testSanitizeMapsPrefixesSplitExpertsAndLoraExperts() throws {
        let model = try Zaya1VL(Self.smallConfig())
        let weights = model.sanitize(weights: [
            "model.embed_tokens.weight": MLXArray.zeros([128, 8]),
            "lm_head.weight": MLXArray.zeros([128, 8]),
            "model.layers.0.zaya_block.experts.switch_mlp.gate_proj.weight":
                MLXArray.zeros([2, 8, 8]),
            "model.layers.0.mlp.zaya_block.experts.local_experts.0.lora_fc1.0.weight":
                MLXArray.zeros([2, 8]),
            "model.layers.0.mlp.zaya_block.experts.local_experts.1.lora_fc1.0.weight":
                MLXArray.zeros([2, 8]),
            "model.layers.0.attn.self_attn.qkv.conv_qk.0.weight": MLXArray.zeros([12, 3, 2]),
            "vision_tower.patch_embed.proj.weight": MLXArray.zeros([8, 3, 1, 14, 14]),
        ])

        XCTAssertNotNil(weights["language_model.model.embed_tokens.weight"])
        XCTAssertNil(weights["language_model.lm_head.weight"])
        XCTAssertEqual(
            weights[
                "language_model.model.layers.0.mlp.zaya_block.experts.switch_mlp.gate_proj.weight",
            ]?.shape,
            [2, 8, 8],
        )
        XCTAssertEqual(
            weights[
                "language_model.model.layers.0.mlp.zaya_block.experts.lora_fc1.0.weight",
            ]?.shape,
            [2, 2, 8],
        )
        XCTAssertEqual(
            weights["language_model.model.layers.0.attn.self_attn.qkv.conv_qk.0.weight"]?
                .shape,
            [12, 2, 3],
        )
        XCTAssertEqual(weights["vision_tower.patch_embed.proj.weight"]?.shape, [8, 1, 14, 14, 3])
    }

    private static func smallConfig() throws -> Zaya1VLConfiguration {
        try JSONDecoder.json5().decode(
            Zaya1VLConfiguration.self,
            from: """
                {
                  "model_type": "zaya1_vl",
                  "attention_bias": false,
                  "cca_time0": 2,
                  "cca_time1": 2,
                  "ffn_hidden_size": 16,
                  "hidden_size": 8,
                  "image_token_id": 99,
                  "max_position_embeddings": 32,
                  "moe_router_topk": 1,
                  "norm_epsilon": 0.00001,
                  "num_attention_heads": 2,
                  "num_experts": 2,
                  "num_hidden_layers": 1,
                  "num_key_value_heads": 1,
                  "num_query_groups": 1,
                  "rope_pct": 0.5,
                  "rotary_base": 10000,
                  "scale_residual_merge": true,
                  "tie_word_embeddings": true,
                  "vision_lora": true,
                  "vision_lora_rank_attn": 2,
                  "vision_lora_rank_mlp": 2,
                  "vocab_size": 128,
                  "zaya_mlp_expansion": 4,
                  "zaya_use_eda": true,
                  "zaya_use_mod": true,
                  "vision_config": {
                    "depth": 1,
                    "hidden_size": 8,
                    "intermediate_size": 12,
                    "in_chans": 3,
                    "model_type": "qwen2_5_vl",
                    "num_heads": 2,
                    "out_hidden_size": 8,
                    "spatial_merge_size": 2,
                    "spatial_patch_size": 14,
                    "temporal_patch_size": 1,
                    "window_size": 28,
                    "fullatt_block_indexes": [0]
                  }
                }
                """.data(using: .utf8)!,
        )
    }
}
