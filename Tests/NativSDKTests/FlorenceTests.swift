import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct FlorenceTestTokenizer: Tokenizer {
    let imageTokenID = 51289

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
        switch text {
        case "<cap>":
            [51269]
        case "<dcap>":
            [51273]
        case "<od>":
            [50265]
        case "<ocr>":
            [50267]
        default:
            text.unicodeScalars.map { Int($0.value) % 1000 + 4 }
        }
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIDs.map { id in
            switch id {
            case imageTokenID: "<florence_image>"
            case 51269: "<cap>"
            case 51273: "<dcap>"
            case 50265: "<od>"
            case 50267: "<ocr>"
            default: "?"
            }
        }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "<florence_image>": imageTokenID
        case "<cap>": 51269
        case "<dcap>": 51273
        case "<od>": 50265
        case "<ocr>": 50267
        default: nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch id {
        case imageTokenID: "<florence_image>"
        case 51269: "<cap>"
        case 51273: "<dcap>"
        case 50265: "<od>"
        case 50267: "<ocr>"
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
        encode(text: "<cap>", addSpecialTokens: false)
    }
}

final class FlorenceTests: XCTestCase {
    func testConfigurationDecodesFlorence2Shape() throws {
        let data = """
            {
              "model_type": "florence2",
              "pad_token_id": 1,
              "text_config": {
                "model_type": "florence2_language",
                "d_model": 768,
                "encoder_attention_heads": 12,
                "decoder_attention_heads": 12,
                "encoder_ffn_dim": 3072,
                "decoder_ffn_dim": 3072,
                "encoder_layers": 6,
                "decoder_layers": 6,
                "decoder_start_token_id": 2,
                "max_position_embeddings": 1024,
                "vocab_size": 51289
              },
              "vision_config": {
                "model_type": "",
                "depths": [1, 1, 9, 1],
                "dim_embed": [128, 256, 512, 1024],
                "num_heads": [4, 8, 16, 32],
                "num_groups": [4, 8, 16, 32],
                "drop_path_rate": 0.1,
                "image_feature_source": ["spatial_avg_pool", "temporal_avg_pool"],
                "image_pos_embed": {"max_pos_embeddings": 50, "type": "learned_abs_2d"},
                "patch_padding": [3, 1, 1, 1],
                "patch_prenorm": [false, true, true, true],
                "patch_size": [7, 3, 3, 3],
                "patch_stride": [4, 2, 2, 2],
                "visual_temporal_embedding": {"max_temporal_embeddings": 100, "type": "COSINE"},
                "window_size": 12
              },
              "vocab_size": 51289
            }
            """.data(using: .utf8)!

        let config = try JSONDecoder.json5().decode(FlorenceConfiguration.self, from: data)

        XCTAssertEqual(config.modelType, "florence2")
        XCTAssertEqual(config.textConfiguration.modelDimension, 768)
        XCTAssertEqual(config.textConfiguration.decoderAttentionHeads, 12)
        XCTAssertEqual(config.textConfiguration.decoderStartTokenID, 2)
        XCTAssertEqual(config.visionConfiguration.embedDimensions.last, 1024)
        XCTAssertEqual(config.visionConfiguration.patchPreNorm, [false, true, true, true])
        XCTAssertEqual(
            config.imageFeatureSource,
            [
                "temporal_avg_pool", "spatial_avg_pool",
            ],
        )
        XCTAssertEqual(config.imageTokenIndex, 51289)
    }

    func testProcessorNormalizesNaturalPromptAndPreprocessesImage() async throws {
        let data = """
            {
              "do_normalize": true,
              "image_mean": [0.485, 0.456, 0.406],
              "image_seq_length": 577,
              "image_std": [0.229, 0.224, 0.225],
              "processor_class": "Florence2Processor",
              "size": {"height": 768, "width": 768}
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder.json5().decode(FlorenceProcessorConfiguration.self, from: data)
        let processor = FlorenceProcessor(config, tokenizer: FlorenceTestTokenizer())
        let image = CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 640, height: 320))
        let input = UserInput(chat: [
            .user("Describe this image in detail.", images: [.ciImage(image)])
        ])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.first, 51289)
        XCTAssertGreaterThan(tokens.count, 10)
        XCTAssertEqual(prepared.image?.pixels.shape, [1, 3, 768, 768])
    }

    func testSanitizeTransposesDaViTConvsAndTiesLMHead() throws {
        let config = try Self.smallConfig()
        let model = Florence(config)
        let weights = model.sanitize(weights: [
            "final_logits_bias": MLXArray.zeros([1, 32]),
            "language_model.model.shared.weight": MLXArray.zeros([32, 16]),
            "vision_tower.convs.0.proj.weight": MLXArray.zeros([16, 3, 7, 7]),
            "vision_tower.blocks.0.0.spatial_block.conv1.fn.dw.weight": MLXArray.zeros([
                16, 1, 3, 3,
            ]),
        ])

        XCTAssertNil(weights["final_logits_bias"])
        XCTAssertEqual(weights["language_model.lm_head.weight"]?.shape, [32, 16])
        XCTAssertEqual(weights["vision_tower.convs.0.proj.weight"]?.shape, [16, 7, 7, 3])
        XCTAssertEqual(
            weights["vision_tower.blocks.0.spatial_block.conv1.fn.dw.weight"]?.shape,
            [16, 3, 3, 1],
        )
    }

    func testPrepareCarriesEncoderStateIntoDecodeStep() throws {
        let config = try Self.smallConfig()
        let model = Florence(config)
        let cache = try model.newCache(configuration: nil)
        let input = LMInput(
            text: .init(tokens: MLXArray([51289, 4, 5]).expandedDimensions(axis: 0)),
        )

        let prepared = try model.prepare(input, cache: cache, state: nil, prefill: .init())
        guard case .tokens(let decoderStart) = prepared else {
            return XCTFail("Expected decoder start token prepare result")
        }
        XCTAssertEqual(decoderStart.tokens.asArray(Int.self), [2, 0])

        let first = model(
            decoderStart,
            cache: cache,
            state: nil,
        )
        XCTAssertEqual(first.logits.shape, [1, 2, 32])
        XCTAssertNotNil(first.state?[florenceCrossAttentionStatesKey])

        let next = model(
            .init(tokens: MLXArray([7]).expandedDimensions(axis: 0)),
            cache: cache,
            state: first.state,
        )
        XCTAssertEqual(next.logits.shape, [1, 1, 32])
        XCTAssertNotNil(next.state?[florenceCrossAttentionStatesKey])
    }

    private static func smallConfig() throws -> FlorenceConfiguration {
        let data = """
            {
              "model_type": "florence2",
              "text_config": {
                "model_type": "florence2_language",
                "d_model": 16,
                "encoder_attention_heads": 2,
                "decoder_attention_heads": 2,
                "encoder_ffn_dim": 32,
                "decoder_ffn_dim": 32,
                "encoder_layers": 1,
                "decoder_layers": 1,
                "decoder_start_token_id": 2,
                "max_position_embeddings": 64,
                "vocab_size": 32
              },
              "vision_config": {
                "depths": [1],
                "dim_embed": [16],
                "num_heads": [2],
                "num_groups": [2],
                "drop_path_rate": 0.0,
                "image_feature_source": ["temporal_avg_pool", "spatial_avg_pool"],
                "image_pos_embed": {"max_pos_embeddings": 50, "type": "learned_abs_2d"},
                "patch_padding": [3],
                "patch_prenorm": [false],
                "patch_size": [7],
                "patch_stride": [4],
                "visual_temporal_embedding": {"max_temporal_embeddings": 100, "type": "COSINE"},
                "window_size": 12
              },
              "vocab_size": 32,
              "image_token_id": 51289
            }
            """.data(using: .utf8)!
        return try JSONDecoder.json5().decode(FlorenceConfiguration.self, from: data)
    }
}
