import CoreImage
import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct LFM2VLAdjacentImageTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [124907, 124907, 1] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { token == "<image>" ? 124907 : nil }
    func convertIdToToken(_ id: Int) -> String? { id == 124907 ? "<image>" : nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [124907, 124907, 1] }
}

final class LFM2Tests: XCTestCase {
    func testAdjacentImagePlaceholdersExpandIndependently() async throws {
        let config = try JSONDecoder().decode(
            LFM2VLProcessorConfiguration.self,
            from: Data(#"{"tile_size":8,"encoder_patch_size":2,"downsample_factor":2}"#.utf8))
        let processor = LFM2VLProcessor(config, tokenizer: LFM2VLAdjacentImageTokenizer())
        let image = CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 8, height: 8))
        let input = try await processor.prepare(
            input: UserInput(chat: [
                .user("Read both images.", images: [.ciImage(image), .ciImage(image)])
            ]))
        XCTAssertEqual(
            input.text.tokens.flattened().asArray(Int.self),
            Array(repeating: 124907, count: 8) + [1])
        XCTAssertEqual(input.image?.pixels.shape, [2, 16, 12])
        XCTAssertEqual(input.image?.frames?.count, 2)
    }

    func testVisionIntermediateSizeLoadsCheckpointAndPreservesExplicitBlockWidth() throws {
        func model(_ widths: [String: Int]) throws -> LFM2VL {
            let text: [String: Any] = [
                "model_type": "lfm2", "hidden_size": 64, "num_hidden_layers": 1,
                "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 32,
                "block_auto_adjust_ff_dim": false,
            ]
            let config: [String: Any] = [
                "model_type": "lfm2_vl",
                "text_config": text.merging(widths.mapValues { $0 as Any }) { _, new in new },
                "vision_config": [
                    "model_type": "siglip2_vision_model", "hidden_size": 16,
                    "intermediate_size": 32, "num_hidden_layers": 1,
                    "num_attention_heads": 2, "num_patches": 4, "patch_size": 2,
                ],
                "projector_hidden_size": 32,
            ]
            return try LFM2VL(
                JSONDecoder().decode(
                    LFM2VLConfiguration.self, from: JSONSerialization.data(withJSONObject: config)))
        }
        let source = try model(["block_ff_dim": 128])
        let weights = source.parameters()
        for widths in [
            ["intermediate_size": 128],
            ["intermediate_size": 256, "block_ff_dim": 128],
        ] {
            let loaded = try model(widths)
            try loaded.update(parameters: weights, verify: .all)
        }
    }

    func testVisionDecoderConfigurationPreservesLayerTypesAndNestedRoPE() throws {
        let json =
            #"{"model_type":"lfm2","hidden_size":64,"num_hidden_layers":4,"num_attention_heads":4,"num_key_value_heads":2,"vocab_size":128,"intermediate_size":192,"conv_L_cache":3,"layer_types":["conv","full_attention","conv","full_attention"],"rope_theta":10000,"rope_parameters":{"rope_theta":1000000}}"#
        let text = try JSONDecoder().decode(
            LFM2VLConfiguration.TextConfiguration.self, from: Data(json.utf8))
        let decoder = LFM2Configuration(text)
        XCTAssertEqual(decoder.fullAttnIdxs, [1, 3])
        XCTAssertEqual(decoder.ropeTheta, 1_000_000)
        XCTAssertEqual(decoder.blockFFDim, 192)
        XCTAssertEqual(decoder.convLCache, 3)
    }

    func testIntermediateSizeLoadsCheckpointAndPreservesExplicitBlockWidth() throws {
        let base: [String: Any] = [
            "model_type": "lfm2", "hidden_size": 64, "num_hidden_layers": 1,
            "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 32,
            "norm_eps": 1e-5, "block_auto_adjust_ff_dim": false,
        ]
        func model(_ widths: [String: Int]) throws -> LFM2Model {
            let values = base.merging(widths.mapValues { $0 as Any }) { _, new in new }
            return try LFM2Model(
                JSONDecoder().decode(
                    LFM2Configuration.self, from: JSONSerialization.data(withJSONObject: values)))
        }
        let source = try model(["block_ff_dim": 128])
        let weights = source.parameters()
        let tokens = MLXArray([1, 2, 3]).reshaped(1, 3)
        let expected = source(tokens, cache: nil)
        for widths in [
            ["intermediate_size": 128],
            ["intermediate_size": 256, "block_ff_dim": 128],
        ] {
            let loaded = try model(widths)
            try loaded.update(parameters: weights, verify: .all)
            XCTAssertTrue(allClose(loaded(tokens, cache: nil), expected).item(Bool.self))
        }
    }
}
