import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct FalconPerceptionTestTokenizer: Tokenizer {
    var bosToken: String? {
        nil
    }

    var eosToken: String? {
        "<|end_of_text|>"
    }

    var unknownToken: String? {
        nil
    }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        var output: [Int] = []
        var index = text.startIndex
        while index < text.endIndex {
            if text[index...].hasPrefix("<|image|>") {
                output.append(227)
                index = text.index(index, offsetBy: "<|image|>".count)
            } else if text[index...].hasPrefix("<|start_of_query|>") {
                output.append(250)
                index = text.index(index, offsetBy: "<|start_of_query|>".count)
            } else if text[index...].hasPrefix("<|REF_SEG|>") {
                output.append(251)
                index = text.index(index, offsetBy: "<|REF_SEG|>".count)
            } else {
                output.append(Int(text[index].unicodeScalars.first!.value))
                index = text.index(after: index)
            }
        }
        return output
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIDs.map { id in
            switch id {
            case 227: "<|image|>"
            case 244: "<|image_cls|>"
            case 245: "<|image_reg_1|>"
            case 246: "<|image_reg_2|>"
            case 247: "<|image_reg_3|>"
            case 248: "<|image_reg_4|>"
            case 250: "<|start_of_query|>"
            case 251: "<|REF_SEG|>"
            case 230: "<|img_end|>"
            default: String(UnicodeScalar(id)!)
            }
        }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "<|image|>": 227
        case "<|image_cls|>": 244
        case "<|image_reg_1|>": 245
        case "<|image_reg_2|>": 246
        case "<|image_reg_3|>": 247
        case "<|image_reg_4|>": 248
        case "<|start_of_query|>": 250
        case "<|REF_SEG|>": 251
        case "<|img_end|>": 230
        default: nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch id {
        case 227: "<|image|>"
        case 244: "<|image_cls|>"
        case 245: "<|image_reg_1|>"
        case 246: "<|image_reg_2|>"
        case 247: "<|image_reg_3|>"
        case 248: "<|image_reg_4|>"
        case 250: "<|start_of_query|>"
        case 251: "<|REF_SEG|>"
        case 230: "<|img_end|>"
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
        encode(
            text:
                "<|image|>Segment these expressions in the image:<|start_of_query|>cats<|REF_SEG|>",
            addSpecialTokens: false,
        )
    }
}

final class FalconPerceptionTests: XCTestCase {
    func testProcessorBuildsPerceptionPromptAndExpandsImageToken() async throws {
        let data = """
            {
              "spatial_patch_size": 2,
              "min_image_size": 4,
              "max_image_size": 16,
              "min_pixels": 16,
              "max_pixels": 256,
              "img_id": 227,
              "img_end_id": 230,
              "image_cls_token_id": 244,
              "image_reg_1_token_id": 245,
              "image_reg_2_token_id": 246,
              "image_reg_3_token_id": 247,
              "image_reg_4_token_id": 248
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder.json5().decode(
            FalconPerceptionProcessorConfiguration.self,
            from: data,
        )
        let processor = FalconPerceptionProcessor(
            config,
            tokenizer: FalconPerceptionTestTokenizer(),
        )
        let image = CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4))
        let input = UserInput(chat: [.user("cats", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 227 }), 4)
        XCTAssertEqual(tokens.count(where: { $0 == 244 }), 1)
        XCTAssertEqual(tokens.count(where: { $0 == 245 }), 1)
        XCTAssertEqual(tokens.count(where: { $0 == 246 }), 1)
        XCTAssertEqual(tokens.count(where: { $0 == 247 }), 1)
        XCTAssertEqual(tokens.count(where: { $0 == 248 }), 1)
        XCTAssertEqual(tokens.count(where: { $0 == 230 }), 1)
        XCTAssertTrue(tokens.contains(250))
        XCTAssertTrue(tokens.contains(251))
        XCTAssertEqual(prepared.image?.pixels.shape, [1, 4, 4, 3])
        XCTAssertEqual(prepared.image?.frames?.first?.h, 2)
        XCTAssertEqual(prepared.image?.frames?.first?.w, 2)
    }

    func testSmallModelRunsVisionToLanguageAndPerceptionFeedbackPath() throws {
        let model = try FalconPerception(Self.smallConfig())
        let tokens = MLXArray([12, 13, 14, 15, 16, 9, 9, 9, 9, 17]).reshaped(1, 10)
        let input = LMInput(
            text: .init(tokens: tokens, mask: ones(like: tokens).asType(.int8)),
            image: .init(pixels: MLXArray.zeros([1, 4, 4, 3]), frames: [THW(1, 2, 2)]),
        )
        let cache = try model.newCache(configuration: nil)

        let prepared = try model.prepare(input, cache: cache, state: nil, prefill: .init())
        guard case .logits(let output) = prepared else {
            return XCTFail("Expected logits")
        }
        XCTAssertEqual(output.logits.shape, [1, 10, 128])

        let coord = model(
            .init(tokens: MLXArray([21]).reshaped(1, 1)),
            cache: cache,
            state: nil,
        )
        XCTAssertEqual(coord.logits.shape, [1, 1, 128])

        let size = model(
            .init(tokens: MLXArray([22]).reshaped(1, 1)),
            cache: cache,
            state: nil,
        )
        XCTAssertEqual(size.logits.shape, [1, 1, 128])
        XCTAssertEqual(model.detections.count, 1)
    }

    func testSanitizeMapsReferenceKeysAndDropsSegmentationWeights() throws {
        let model = try FalconPerception(Self.smallConfig())
        let w13 = MLXArray(0 ..< 24).reshaped(6, 4)

        let weights = model.sanitize(weights: [
            "tok_embeddings.weight": MLXArray.zeros([32, 8]),
            "img_projector.weight": MLXArray.zeros([8, 12]),
            "coord_encoder.embed.weight": MLXArray.zeros([4, 2]),
            "coord_decoder.w1.weight": MLXArray.zeros([8, 8]),
            "size_encoder.transform.weight": MLXArray.zeros([8, 4]),
            "size_decoder.w2.weight": MLXArray.zeros([8, 8]),
            "output.weight": MLXArray.zeros([32, 8]),
            "freqs_cis_golden": MLXArray.zeros([2, 2, 2]),
            "layers.0.attention.wqkv.weight": MLXArray.zeros([24, 8]),
            "layers.0.feed_forward.w13.weight": w13,
            "proj_segm.pixel_layer.weight": MLXArray.zeros([4, 8]),
            "conv_segm.weight": MLXArray.zeros([4, 3, 3, 8]),
            "itok_upsampler.proj.weight": MLXArray.zeros([4, 4]),
        ])

        XCTAssertEqual(weights["language_model.model.embed_tokens.weight"]?.shape, [32, 8])
        XCTAssertEqual(weights["language_model.model.img_projector.weight"]?.shape, [8, 12])
        XCTAssertEqual(weights["coord_encoder.embed.weight"]?.shape, [4, 2])
        XCTAssertEqual(weights["coord_decoder.w1.weight"]?.shape, [8, 8])
        XCTAssertEqual(weights["size_encoder.transform.weight"]?.shape, [8, 4])
        XCTAssertEqual(weights["size_decoder.w2.weight"]?.shape, [8, 8])
        XCTAssertEqual(weights["language_model.lm_head.weight"]?.shape, [32, 8])
        XCTAssertEqual(weights["language_model.model.freqs_cis_golden"]?.shape, [2, 2, 2])
        XCTAssertEqual(
            weights["language_model.model.layers.0.self_attn.wqkv.weight"]?.shape,
            [24, 8],
        )
        XCTAssertEqual(
            weights["language_model.model.layers.0.mlp.w13.weight"]?.asArray(Int.self),
            [
                0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 4, 5, 6, 7, 12, 13, 14, 15, 20,
                21, 22, 23,
            ],
        )
        XCTAssertNil(weights["proj_segm.pixel_layer.weight"])
        XCTAssertNil(weights["conv_segm.weight"])
        XCTAssertNil(weights["itok_upsampler.proj.weight"])
    }

    private static func smallConfig() throws -> FalconPerceptionConfiguration {
        let data = """
            {
              "model_type": "falcon_perception",
              "dim": 8,
              "n_layers": 1,
              "n_heads": 2,
              "head_dim": 4,
              "n_kv_heads": 1,
              "vocab_size": 128,
              "ffn_dim": 16,
              "norm_eps": 1e-5,
              "max_seq_len": 64,
              "spatial_patch_size": 2,
              "coord_enc_dim": 8,
              "coord_dec_dim": 8,
              "coord_out_dim": 8,
              "coord_token_id": 21,
              "size_enc_dim": 8,
              "size_dec_dim": 8,
              "size_out_dim": 8,
              "size_token_id": 22,
              "seg_token_id": 23,
              "do_segmentation": false,
              "img_id": 9,
              "eos_id": 11,
              "image_cls_token_id": 12,
              "image_reg_1_token_id": 13,
              "image_reg_2_token_id": 14,
              "image_reg_3_token_id": 15,
              "image_reg_4_token_id": 16,
              "img_end_id": 17
            }
            """.data(using: .utf8)!
        return try JSONDecoder.json5().decode(FalconPerceptionConfiguration.self, from: data)
    }
}
