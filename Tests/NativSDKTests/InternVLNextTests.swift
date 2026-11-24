import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

/// The `internvl` family, which is not the `internvl_chat` family InternVL.swift holds.
/// Its tower embeds patches with a plain conv, prepends a class token, resamples its
/// position table off-resolution, and scales each residual branch by a learned lambda.
final class InternVLNextTests: XCTestCase {
    private static func json(vision: String = "", imageSize: String = "32") -> String {
        """
        {
          "model_type": "internvl",
          "downsample_ratio": 0.5,
          "image_token_id": 7,
          "text_config": {
            "model_type": "qwen3", "hidden_size": 32, "num_hidden_layers": 2,
            "intermediate_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6, "vocab_size": 64, "head_dim": 8, "rope_theta": 10000.0,
            "tie_word_embeddings": false
          },
          "vision_config": {
            "model_type": "internvl", "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": 2, "num_attention_heads": 2,
            "image_size": \(imageSize), "patch_size": 8, "layer_norm_eps": 1e-6
            \(vision)
          }
        }
        """
    }

    private func configuration(_ json: String) throws -> InternVLNextConfiguration {
        try JSONDecoder().decode(InternVLNextConfiguration.self, from: Data(json.utf8))
    }

    /// The shuffle folds each 2x2 block of patches into the channel axis: 16 patches of
    /// 16 channels become 4 of 64.
    func testPixelShuffleFoldsNeighbourhoodsIntoChannels() {
        let input = MLXArray(0 ..< Int32(1 * 16 * 16)).reshaped(1, 16, 16).asType(.float32)
        let out = InternVLNext.pixelShuffle(input, downsampleRatio: 0.5)
        XCTAssertEqual(out.shape, [1, 4, 64])
    }

    /// A conv exported channels-first has to be transposed; one already in MLX's layout
    /// must be left alone.
    func testPatchConvIsTransposedOnlyWhenChannelsFirst() throws {
        let model = InternVLNext(try configuration(Self.json()))
        let key = "vision_tower.embeddings.patch_embeddings.projection.weight"
        let nchw = model.sanitize(weights: [key: MLXArray.zeros([16, 3, 8, 8])])
        XCTAssertEqual(nchw[key]?.shape, [16, 8, 8, 3])
        let nhwc = model.sanitize(weights: [key: MLXArray.zeros([16, 8, 8, 3])])
        XCTAssertEqual(nhwc[key]?.shape, [16, 8, 8, 3])
    }

    /// Some conversions nest the backbone one level too deep.
    func testDoubledModelPrefixIsCollapsed() throws {
        let model = InternVLNext(try configuration(Self.json()))
        let out = model.sanitize(weights: [
            "language_model.model.model.norm.weight": MLXArray.ones([32]),
            "language_model.model.lm_head.weight": MLXArray.zeros([64, 32]),
        ])
        XCTAssertNotNil(out["language_model.model.norm.weight"])
        XCTAssertNotNil(out["language_model.lm_head.weight"])
        XCTAssertNil(out["language_model.model.model.norm.weight"])
    }

    /// Off-resolution input resamples the position table rather than failing.
    func testPositionTableIsResampledOffResolution() throws {
        let configuration = try configuration(Self.json())
        let tower = InternVLNextVision.VisionModel(configuration.visionConfiguration)
        let out = tower(MLXArray.zeros([1, 16, 16, 3]))
        XCTAssertEqual(out.shape, [1, (16 / 8) * (16 / 8) + 1, 16])
    }
}
