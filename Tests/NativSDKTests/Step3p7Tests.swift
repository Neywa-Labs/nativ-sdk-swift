import CoreImage
import Foundation
import MLX
import XCTest

@testable import NativSDK

private final class PrefillProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var updates: [[Int]] = []

    func record(_ processed: Int, _ total: Int) {
        lock.withLock { updates.append([processed, total]) }
    }

    var values: [[Int]] {
        lock.withLock { updates }
    }
}

private struct Step3p7TestTokenizer: Tokenizer {
    private let tokenIDs: [String: Int] = [
        "<｜begin▁of▁sentence｜>": 0,
        "<｜end▁of▁sentence｜>": 1,
        "<|im_start|>": 10,
        "<|im_end|>": 11,
        "<im_start>": 12,
        "<im_end>": 13,
        "<patch_start>": 14,
        "<patch_end>": 15,
        "<patch_newline>": 16,
        "<im_patch>": 31,
    ]

    var bosToken: String? {
        "<｜begin▁of▁sentence｜>"
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
                result.append(Int(text[index].unicodeScalars.first!.value % 32) + 32)
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

final class Step3p7Tests: XCTestCase {
    func testConfigurationDecodesCheckpointShape() throws {
        let config = try JSONDecoder.json5().decode(
            Step3p7Configuration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "step3p7",
                  "image_token_id": 128001,
                  "projector_bias": false,
                  "text_config": {
                    "model_type": "step3p5",
                    "vocab_size": 128896,
                    "hidden_size": 4096,
                    "intermediate_size": 11264,
                    "num_hidden_layers": 45,
                    "num_attention_heads": 64,
                    "num_attention_groups": 8,
                    "moe_intermediate_size": 1280,
                    "moe_num_experts": 288,
                    "moe_top_k": 8,
                    "share_expert_dims": 1280,
                    "head_dim": 128,
                    "norm_expert_weight": true,
                    "rope_theta": [5000000.0, 10000.0],
                    "sliding_window": 512,
                    "layer_types": ["full_attention", "sliding_attention", "sliding_attention"],
                    "partial_rotary_factors": [0.5, 1.0],
                    "moe_layers_enum": "3,4,5,44",
                    "use_head_wise_attn_gate": true,
                    "attention_other_setting": {
                      "attention_type": "sliding_attention",
                      "num_attention_heads": 96,
                      "num_attention_groups": 8,
                      "head_dim": 128
                    },
                    "eos_token_id": [1, 2, 128007],
                    "bos_token_id": 0
                  },
                  "vision_config": {
                    "model_type": "perception_encoder",
                    "image_size": 728,
                    "patch_size": 14,
                    "width": 1536,
                    "layers": 47,
                    "heads": 16,
                    "use_cls_token": false,
                    "ls_init_value": 0.1
                  }
                }
                """.data(using: .utf8)),
        )

        XCTAssertEqual(config.modelType, "step3p7")
        XCTAssertEqual(config.imageTokenIndex, 128_001)
        XCTAssertEqual(config.textConfiguration.modelType, "step3p5")
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 45)
        XCTAssertEqual(config.textConfiguration.ropeTheta[0], 5_000_000)
        XCTAssertEqual(config.textConfiguration.ropeTheta[1], 10000)
        XCTAssertFalse(config.textConfiguration.isSliding(layer: 0))
        XCTAssertTrue(config.textConfiguration.isSliding(layer: 1))
        XCTAssertTrue(config.textConfiguration.moeLayers.contains(44))
        XCTAssertEqual(config.textConfiguration.attentionOtherSetting?.attentionHeads, 96)
        XCTAssertEqual(config.visionConfiguration.modelType, "perception_encoder")
        XCTAssertEqual(config.visionConfiguration.width, 1536)
        XCTAssertEqual(config.visionConfiguration.layers, 47)
        XCTAssertEqual(config.visionConfiguration.layerScaleInitValue, 0.1)
    }

    func testProcessorBuildsBaseAndPatchTensorsAndExpandsImageTokens() async throws {
        let config = try JSONDecoder.json5().decode(
            Step3VLProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "Step3VLProcessor",
                  "image_size": 28,
                  "patch_image_size": 14,
                  "max_image_size": 128,
                  "num_image_feature_size": 1,
                  "num_patch_feature_size": 1
                }
                """.data(using: .utf8)),
        )
        let processor = Step3VLProcessor(config, tokenizer: Step3p7TestTokenizer())
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 40, height: 40))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 31 }), 10)
        XCTAssertEqual(prepared.image?.pixels.shape, [1, 3, 28, 28])
        XCTAssertEqual(prepared.image?.frames?.first?.t, 9)
        XCTAssertEqual(prepared.video?.pixels.shape, [9, 3, 14, 14])
        XCTAssertEqual(prepared.video?.frames?.first?.t, 9)
        XCTAssertTrue(tokens.contains(14))
        XCTAssertTrue(tokens.contains(15))
        XCTAssertTrue(tokens.contains(16))
    }

    func testSmallModelPrepareRunsVisionToLanguagePath() throws {
        let model = try Step3p7(Self.smallConfig())
        let inputIDs = MLXArray([7, 31, 8]).reshaped(1, 3)
        let pixels = MLXArray.zeros([1, 3, 28, 28])
        let input = LMInput(
            text: .init(tokens: inputIDs),
            image: .init(pixels: pixels, frames: [THW(0, 0, 0)]),
        )
        let result = try model.prepare(
            input, cache: model.newCache(configuration: nil), state: nil,
            prefill: .init(stepSize: 2))

        guard case .logits(let output) = result else {
            return XCTFail("Expected logits from prepare")
        }
        XCTAssertEqual(output.logits.shape, [1, 1, 64])
    }

    func testPrepareHonorsPrefillChunkingAndReportsProgress() throws {
        MLXRandom.seed(0)
        let model = try Step3p7(Self.smallConfig())
        let input = LMInput(tokens: MLXArray([7, 8, 9, 10]).reshaped(1, 4))
        let chunkedProgress = PrefillProgressRecorder()
        let chunkedCache = try model.newCache(configuration: nil)
        let chunked = try model.prepare(
            input, cache: chunkedCache, state: nil,
            prefill: .init(stepSize: 2, progress: { chunkedProgress.record($0, $1) }),
        )
        let unchunkedProgress = PrefillProgressRecorder()
        let unchunkedCache = try model.newCache(configuration: nil)
        let unchunked = try model.prepare(
            input, cache: unchunkedCache, state: nil,
            prefill: .init(
                stepSize: 2, chunking: .unchunked,
                progress: { unchunkedProgress.record($0, $1) }),
        )

        guard case .logits(let chunkedOutput) = chunked,
            case .logits(let unchunkedOutput) = unchunked
        else {
            return XCTFail("Expected logits from prepare")
        }
        XCTAssertEqual(chunkedOutput.logits.shape, [1, 1, 64])
        XCTAssertEqual(unchunkedOutput.logits.shape, [1, 4, 64])
        XCTAssertEqual(chunkedProgress.values, [[2, 4], [3, 4], [4, 4]])
        XCTAssertEqual(unchunkedProgress.values, [[4, 4]])
        XCTAssertTrue(chunkedCache.allSatisfy { $0.offset == 4 })
        XCTAssertTrue(unchunkedCache.allSatisfy { $0.offset == 4 })
        // Different forward shapes select different GPU kernels; allow their rounding drift.
        XCTAssertTrue(
            allClose(
                chunkedOutput.logits[0..., -1, 0...],
                unchunkedOutput.logits[0..., -1, 0...], rtol: 5e-3, atol: 5e-3
            ).item(Bool.self))
    }

    func testSanitizerMapsReferenceAndCheckpointKeys() throws {
        let model = try Step3p7(Self.smallConfig())
        let weights = model.sanitize(weights: [
            "model.vision_model.conv1.weight": MLXArray.zeros([4, 3, 14, 14]),
            "vision_model.transformer.resblocks.0.attn.in_proj_weight": MLXArray.zeros([12, 4]),
            "model.vit_large_projector.weight": MLXArray.zeros([8, 16]),
            "model.layers.1.moe.gate_proj.weight": MLXArray.zeros([2, 4, 8]),
            "model.layers.3.input_layernorm.weight": MLXArray.zeros([8]),
            "lm_head.weight": MLXArray.zeros([64, 8]),
        ])

        XCTAssertEqual(weights["vision_model.conv1.weight"]?.shape, [4, 14, 14, 3])
        XCTAssertEqual(weights["vision_model.transformer.0.attn.in_proj.weight"]?.shape, [12, 4])
        XCTAssertEqual(weights["vit_large_projector.weight"]?.shape, [8, 16])
        XCTAssertEqual(
            weights["language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape,
            [2, 4, 8],
        )
        XCTAssertNil(weights["language_model.model.layers.3.input_layernorm.weight"])
        XCTAssertEqual(weights["language_model.lm_head.weight"]?.shape, [64, 8])
    }

    private static func smallConfig() throws -> Step3p7Configuration {
        try JSONDecoder.json5().decode(
            Step3p7Configuration.self,
            from: """
                {
                  "model_type": "step3p7",
                  "image_token_id": 31,
                  "vocab_size": 64,
                  "projector_bias": false,
                  "text_config": {
                    "model_type": "step3p5",
                    "vocab_size": 64,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_attention_groups": 2,
                    "moe_intermediate_size": 4,
                    "moe_num_experts": 2,
                    "moe_top_k": 1,
                    "share_expert_dim": 4,
                    "head_dim": 4,
                    "rope_theta": [10000.0, 10000.0],
                    "sliding_window": 4,
                    "layer_types": ["full_attention", "sliding_attention"],
                    "partial_rotary_factors": [1.0, 1.0],
                    "moe_layers_enum": [1],
                    "use_head_wise_attn_gate": true,
                    "attention_other_setting": {
                      "num_attention_heads": 2,
                      "num_attention_groups": 2,
                      "head_dim": 4
                    },
                    "eos_token_id": [1, 2],
                    "bos_token_id": 0
                  },
                  "vision_config": {
                    "model_type": "perception_encoder",
                    "image_size": 28,
                    "patch_size": 14,
                    "width": 4,
                    "layers": 1,
                    "heads": 1,
                    "mlp_ratio": 2.0,
                    "use_cls_token": false,
                    "ls_init_value": 0.1
                  }
                }
                """.data(using: .utf8)!,
        )
    }
}
