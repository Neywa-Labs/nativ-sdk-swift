import Foundation
import MLX
import XCTest

@testable import NativSDK

final class MOSSImageTests: XCTestCase {
    func testFlowScheduleSnapsAndDeduplicates() throws {
        let config = try MOSSImageScheduleConfiguration(.init())
        let standard = MOSSImageSchedule(steps: 50, configuration: config)
        XCTAssertEqual(standard.timesteps.count, 50)
        XCTAssertEqual(standard.timesteps[0], 1000)
        XCTAssertEqual(standard.timesteps[49], 24)
        XCTAssertTrue(zip(standard.timesteps, standard.timesteps.dropFirst()).allSatisfy(>))
        XCTAssertEqual(standard.sigmas.last, 0)
        let dense = MOSSImageSchedule(steps: 2000, configuration: config)
        XCTAssertLessThanOrEqual(dense.timesteps.count, 1000)
        XCTAssertTrue(zip(dense.timesteps, dense.timesteps.dropFirst()).allSatisfy(>))
        XCTAssertEqual(MOSSImageSchedule(steps: 1, configuration: config).sigmas, [1, 0])
    }

    func testLayerGuidanceUsesHalfOpenRangeAndValidatesIndices() throws {
        XCTAssertEqual(try MOSSImageLayerGuidance(.init(), depth: 42).unconditionalLayers, [22])
        let options = MOSSImageGenerationOptions(
            unconditionalSkipLayers: [], skipLayerGuidanceLayers: [0, 2, 2],
            skipLayerGuidanceScale: 1.5, skipLayerGuidanceStart: 0.25, skipLayerGuidanceStop: 0.75)
        let guidance = try MOSSImageLayerGuidance(options, depth: 3)
        XCTAssertTrue(guidance.unconditionalLayers.isEmpty)
        XCTAssertEqual(guidance.positiveLayers, [0, 2])
        XCTAssertEqual(
            (0 ..< 4).map { guidance.isActive(step: $0, count: 4) }, [false, true, true, false])
        XCTAssertThrowsError(try MOSSImageLayerGuidance(options, depth: 2))
    }

    func testNativeTextWeightsSkipVisionAndSelectOffsetNorms() {
        let weights = MOSSImagePipeline.sanitizeTextEncoder([
            "visual.patch_embed.weight": MLXArray(7),
            "language_model.norm.weight": MLXArray([Float(0.25)]),
            "language_model.layers.0.linear_attn.norm.weight": MLXArray([Float(0.75)]),
            "language_model.layers.0.linear_attn.conv1d.weight": MLXArray.zeros([32, 1, 4]),
        ])
        XCTAssertEqual(weights.count, 3)
        XCTAssertEqual(weights["norm.weight"]?.asArray(Float.self), [1.25])
        XCTAssertEqual(weights["layers.0.linear_attn.norm.weight"]?.asArray(Float.self), [0.75])
        XCTAssertEqual(weights["layers.0.linear_attn.conv1d.weight"]?.shape, [32, 4, 1])
    }
}
