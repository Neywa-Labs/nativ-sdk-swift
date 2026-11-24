import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class MiMoV2FlashTests: XCTestCase {
    private func configuration(valueScale: Float? = nil) throws -> MiMoV2FlashConfiguration {
        var json =
            try JSONSerialization.jsonObject(
                with: ModelTestConfigurations.textConfiguration("mimo_v2_flash")) as! [String: Any]
        if let valueScale { json["attention_value_scale"] = valueScale }
        return try JSONDecoder().decode(
            MiMoV2FlashConfiguration.self, from: JSONSerialization.data(withJSONObject: json))
    }

    func testFP8BlockScalingBeforeExpertStacking() throws {
        var config = try configuration()
        config.nRoutedExperts = 2
        let model = MiMoV2FlashModel(config)
        let prefix = "model.layers.0.mlp"
        var weights: [String: MLXArray] = [:]
        // E4M3 encodes +1 as 0x38 and -2 as 0xc0. Both dimensions cross a block
        // boundary and require padding, with a distinct scale for each block.
        for (expert, byte) in [UInt8(0x38), UInt8(0xc0)].enumerated() {
            let key = "\(prefix).experts.\(expert).gate_proj.weight"
            weights[key] = MLXArray(Array(repeating: byte, count: 129 * 130), [129, 130])
            weights[key + "_scale_inv"] = MLXArray([Float(1), 2, 4, 8], [2, 2])
        }
        let sanitized = model.sanitize(weights: weights)
        let actual = try XCTUnwrap(sanitized["\(prefix).switch_mlp.gate_proj.weight"])
        let expected: [Float] = [Float(1), -2].flatMap { value in
            (0 ..< 129).flatMap { row in
                (0 ..< 130).map { column in
                    value * [Float(1), 2, 4, 8][(row / 128) * 2 + column / 128]
                }
            }
        }
        XCTAssertEqual(actual.shape, [2, 129, 130])
        XCTAssertEqual(actual.dtype, .bfloat16)
        XCTAssertEqual(actual.asArray(Float.self), expected)
        XCTAssertFalse(sanitized.keys.contains { $0.hasSuffix("weight_scale_inv") })
    }

    func testAttentionValueScalingMatchesCausalMeanAndCachedDecode() throws {
        for sliding in [false, true] {
            for valueScale: Float? in [nil, 0.5] {
                var config = try configuration(valueScale: valueScale)
                config.hiddenSize = 4
                config.attentionHeads = 2
                config.kvHeads = 2
                config.headDim = 2
                config.vHeadDim = 2
                config.swaAttentionHeads = 2
                config.swaKvHeads = 2
                config.swaHeadDim = 2
                config.swaVHeadDim = 2
                config.addSwaAttentionSinkBias = false
                config.addFullAttentionSinkBias = false
                let attention = MiMoV2FlashAttention(config, isSlidingWindow: sliding)
                try attention.update(
                    parameters: ModuleParameters.unflattened([
                        "q_proj.weight": MLXArray.zeros([4, 4]),
                        "k_proj.weight": MLXArray.zeros([4, 4]),
                        "v_proj.weight": MLXArray.eye(4),
                        "o_proj.weight": MLXArray.eye(4),
                    ]), verify: .noUnusedKeys)
                let input = MLXArray([Float(1), 2, 3, 4, 5, 8, 11, 14], [1, 2, 4])
                let expected =
                    MLXArray([Float(1), 2, 3, 4, 3, 5, 7, 9], [1, 2, 4])
                    * (valueScale ?? 1)
                let full = attention(input, mask: .causal, cache: nil)
                XCTAssertTrue(allClose(full, expected).item(Bool.self))

                let cache: any KVCache =
                    sliding
                    ? RotatingKVCache(maxSize: 8) : KVCacheSimple()
                let first = attention(input[0..., ..<1, 0...], mask: .none, cache: cache)
                let next = attention(input[0..., 1..., 0...], mask: .none, cache: cache)
                XCTAssertTrue(
                    allClose(concatenated([first, next], axis: 1), expected).item(Bool.self))
            }
        }
    }
}
