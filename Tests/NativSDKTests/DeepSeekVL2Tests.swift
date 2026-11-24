import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct DeepSeekVL2TestTokenizer: Tokenizer {
    let imageTokenID = 100_015
    let userTokenID = 100_021
    let assistantTokenID = 100_022

    var bosToken: String? {
        "<｜begin▁of▁sentence｜>"
    }

    var eosToken: String? {
        "<｜end▁of▁sentence｜>"
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
            } else if text[index...].hasPrefix("<|User|>") {
                output.append(userTokenID)
                index = text.index(index, offsetBy: "<|User|>".count)
            } else if text[index...].hasPrefix("<|Assistant|>") {
                output.append(assistantTokenID)
                index = text.index(index, offsetBy: "<|Assistant|>".count)
            } else {
                output.append(Int(text[index].unicodeScalars.first!.value))
                index = text.index(after: index)
            }
        }
        return output
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIDs.map { id in
            if id == imageTokenID {
                return "<image>"
            }
            if id == userTokenID {
                return "<|User|>"
            }
            if id == assistantTokenID {
                return "<|Assistant|>"
            }
            return UnicodeScalar(id).map(String.init) ?? "?"
        }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "<image>": imageTokenID
        case "<|User|>": userTokenID
        case "<|Assistant|>": assistantTokenID
        case bosToken: 0
        case eosToken: 1
        default: nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch id {
        case imageTokenID: "<image>"
        case userTokenID: "<|User|>"
        case assistantTokenID: "<|Assistant|>"
        case 0: bosToken
        case 1: eosToken
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

final class DeepSeekVL2Tests: XCTestCase {
    func testConfigurationDecodesTinyShape() throws {
        let config = try JSONDecoder.json5().decode(
            DeepSeekVL2Configuration.self,
            from: Self.configJSON(),
        )

        XCTAssertEqual(config.modelType, "deepseek_vl_v2")
        XCTAssertEqual(config.textConfiguration.modelType, "deepseek_v2")
        XCTAssertEqual(config.textConfiguration.hiddenSize, 8)
        XCTAssertEqual(config.textConfiguration.kvHeads, 2)
        XCTAssertFalse(config.textConfiguration.usesMLA)
        XCTAssertEqual(config.visionConfiguration.hiddenSize, 8)
        XCTAssertEqual(config.visionConfiguration.intermediateSize, 16)
        XCTAssertEqual(config.visionConfiguration.patchSize, 14)
        XCTAssertEqual(config.projectorConfiguration.projectorType, "downsample_mlp_gelu")
        XCTAssertEqual(config.projectorConfiguration.downsampleRatio, 2)
        XCTAssertEqual(config.imageTokenIndex, 128_815)
        XCTAssertEqual(config.candidateResolutions.first, [28, 28])
    }

    func testProcessorBuildsGlobalAndLocalTokenRun() async throws {
        let config = try JSONDecoder.json5().decode(
            DeepSeekVL2ProcessorConfiguration.self,
            from: Self.processorJSON(),
        )
        let processor = DeepSeekVL2Processor(config, tokenizer: DeepSeekVL2TestTokenizer())
        let input = UserInput(chat: [
            .user("Describe.", images: [.ciImage(Self.image(width: 20, height: 12))])
        ])

        let prepared = try await processor.prepare(input: input)

        XCTAssertEqual(prepared.image?.pixels.shape, [1, 2, 3, 28, 28])
        XCTAssertEqual(prepared.video?.pixels.shape, [1, 1, 2])
        XCTAssertEqual(prepared.video?.pixels.asArray(Int.self), [1, 1])
        XCTAssertEqual(prepared.text.tokens.asArray(Int.self).first, 0)
        XCTAssertEqual(prepared.text.mask?.asArray(Int8.self).count(where: { $0 == 1 }), 5)

        let tokens = prepared.text.tokens.asArray(Int.self)
        XCTAssertTrue(tokens.contains(DeepSeekVL2TestTokenizer().userTokenID))
        XCTAssertTrue(tokens.contains(DeepSeekVL2TestTokenizer().assistantTokenID))
    }

    func testSanitizeMapsKeysAndStacksExperts() throws {
        let model = try DeepSeekVL2(
            JSONDecoder.json5().decode(
                DeepSeekVL2Configuration.self,
                from: Self.configJSON(),
            ),
        )
        let weights = model.sanitize(weights: [
            "language.model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([8, 8]),
            "vision.patch_embed.proj.weight": MLXArray.zeros([8, 3, 14, 14]),
            "view_seperator": MLXArray.zeros([8]),
            "language_model.model.layers.1.mlp.experts.0.gate_proj.weight":
                MLXArray.zeros([4, 8]),
            "language_model.model.layers.1.mlp.experts.1.gate_proj.weight":
                MLXArray.zeros([4, 8]),
        ])

        XCTAssertNotNil(weights["language_model.model.layers.0.self_attn.q_proj.weight"])
        XCTAssertEqual(
            weights["vision.vision_tower.patch_embed.proj.weight"]?.shape, [8, 14, 14, 3])
        XCTAssertNotNil(weights["view_separator"])
        XCTAssertNil(weights["view_seperator"])
        XCTAssertEqual(
            weights["language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape,
            [2, 4, 8])
    }

    private static func image(width: Int, height: Int) -> CIImage {
        CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
    }

    private static func processorJSON() -> Data {
        """
        {
          "processor_class": "DeepseekVLV2Processor",
          "candidate_resolutions": [[28, 28]],
          "patch_size": 14,
          "downsample_ratio": 2,
          "image_mean": [0.5, 0.5, 0.5],
          "image_std": [0.5, 0.5, 0.5],
          "normalize": true,
          "image_token": "<image>"
        }
        """.data(using: .utf8)!
    }

    private static func configJSON() -> Data {
        """
        {
          "model_type": "deepseek_vl_v2",
          "image_token_index": 100015,
          "tile_tag": "2D",
          "global_view_pos": "head",
          "candidate_resolutions": [[28, 28]],
          "language_config": {
            "model_type": "deepseek_v2",
            "vocab_size": 128,
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": 4,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "n_shared_experts": 1,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "q_lora_rank": null,
            "kv_lora_rank": null,
            "qk_nope_head_dim": 0,
            "qk_rope_head_dim": 0,
            "v_head_dim": 0
          },
          "vision_config": {
            "model_type": "vision",
            "model_name": "siglip_so400m_patch14_384",
            "width": 8,
            "intermediate_size": 16,
            "layers": 1,
            "num_attention_heads": 2,
            "image_size": 28,
            "patch_size": 14,
            "num_channels": 3,
            "mlp_ratio": 2.0
          },
          "projector_config": {
            "model_type": "mlp_projector",
            "projector_type": "downsample_mlp_gelu",
            "input_dim": 8,
            "n_embed": 8,
            "depth": 2,
            "mlp_ratio": 1,
            "downsample_ratio": 2
          }
        }
        """.data(using: .utf8)!
    }
}
