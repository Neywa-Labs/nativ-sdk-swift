import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct YoutuVLTestTokenizer: Tokenizer {
    private let tokenIDs: [String: Int] = [
        "<|begin_of_text|>": 10,
        "<|end_of_text|>": 11,
        "<|vision_start|>": 12,
        "<|vision_end|>": 13,
        "<|image_pad|>": 99,
    ]

    var bosToken: String? {
        "<|begin_of_text|>"
    }

    var eosToken: String? {
        "<|end_of_text|>"
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

final class YoutuVLTests: XCTestCase {
    func testProcessorPacksSiglip2PatchesAndExpandsImageTokens() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            YoutuVLProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "YoutuVLProcessor",
                  "patch_size": 2,
                  "merge_size": 2,
                  "max_image_patches": 4,
                  "image_mean": [0.5, 0.5, 0.5],
                  "image_std": [0.5, 0.5, 0.5]
                }
                """.data(using: .utf8)),
        )
        let processor = YoutuVLProcessor(processorConfig, tokenizer: YoutuVLTestTokenizer())
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)

        XCTAssertEqual(prepared.image?.pixels.shape, [4, 12])
        XCTAssertEqual(prepared.image?.frames?.first?.h, 2)
        XCTAssertEqual(prepared.image?.frames?.first?.w, 2)
        let tokens = prepared.text.tokens.asArray(Int.self)
        XCTAssertEqual(tokens.count(where: { $0 == 99 }), 1)
        XCTAssertTrue(tokens.contains(12))
        XCTAssertTrue(tokens.contains(13))
    }

    func testReducedModelPrepareRunsVisionToLanguagePath() throws {
        let model = try YoutuVL(Self.smallConfig())
        let input = LMInput(
            text: .init(tokens: MLXArray([99]).expandedDimensions(axis: 0)),
            image: .init(
                pixels: zeros([4, 12]),
                frames: [THW(1, 2, 2)],
            ),
        )
        let cache = try model.newCache(configuration: nil)

        let result = try model.prepare(input, cache: cache, state: nil, prefill: .init())
        guard case .logits(let output) = result else {
            return XCTFail("Expected logits")
        }
        XCTAssertEqual(output.logits.shape, [1, 1, 128])
    }

    func testSanitizerMapsOfficialCheckpointKeys() throws {
        let model = try YoutuVL(Self.smallConfig())
        let weights = model.sanitize(weights: [
            "siglip2.vision_model.embeddings.patch_embedding.weight": zeros([8, 12]),
            "siglip2.vision_model.encoder.layers.0.self_attn.q_proj.weight": zeros([8, 8]),
            "merger.ln_q.weight": zeros([8]),
            "model.layers.0.self_attn.kv_b_proj.weight": zeros([16, 4]),
            "model.embed_tokens.weight": zeros([128, 16]),
            "lm_head.weight": zeros([128, 16]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": zeros([2]),
        ])

        XCTAssertNotNil(weights["vision_tower.embeddings.patch_embedding.weight"])
        XCTAssertNotNil(weights["vision_tower.encoder.layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(weights["vision_tower.merger.ln_q.weight"])
        XCTAssertNotNil(weights["language_model.model.layers.0.self_attn.kv_b_proj.weight"])
        XCTAssertNotNil(weights["language_model.model.embed_tokens.weight"])
        XCTAssertNil(weights["language_model.lm_head.weight"])
        XCTAssertNil(weights["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
    }

    private static func smallConfig() throws -> YoutuVLConfiguration {
        try JSONDecoder.json5().decode(
            YoutuVLConfiguration.self,
            from: """
                {
                  "model_type": "youtu_vl",
                  "image_token_id": 99,
                  "video_token_id": 100,
                  "vocab_size": 128,
                  "eos_token_id": 11,
                  "text_config": {
                    "model_type": "youtu_vl",
                    "vocab_size": 128,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "max_position_embeddings": 64,
                    "rms_norm_eps": 0.000001,
                    "kv_lora_rank": 4,
                    "q_lora_rank": 4,
                    "qk_rope_head_dim": 4,
                    "qk_nope_head_dim": 4,
                    "v_head_dim": 4,
                    "rope_theta": 10000,
                    "rope_interleave": true,
                    "tie_word_embeddings": true
                  },
                  "vision_config": {
                    "model_type": "siglip2_vision_model",
                    "hidden_size": 8,
                    "out_hidden_size": 16,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_channels": 3,
                    "num_patches": 4,
                    "patch_size": 2,
                    "spatial_merge_size": 2,
                    "window_size": 4,
                    "fullatt_block_indexes": []
                  }
                }
                """.data(using: .utf8)!,
        )
    }
}
