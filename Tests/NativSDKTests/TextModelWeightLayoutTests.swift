import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class TextModelWeightLayoutTests: XCTestCase {
    func testSeparateExpertsAndConvolutionLayoutsPreserveLogits() throws {
        for (name, compressed) in [
            ("mellum", false), ("exaone_moe", false), ("exaone_moe", true), ("mamba", false),
        ] {
            let data = try ModelTestConfigurations.configuration(name)
            let original = try ModelTestConfigurations.model(name)
            let restored = try ModelRegistry.default.createModel(
                configuration: data, modelType: name)
            if compressed {
                for model in [original, restored] {
                    quantize(model: model) { _, module in
                        guard
                            let weight = module.parameters().flattened().first(where: {
                                $0.0 == "weight"
                            })?.1,
                            weight.dim(-1).isMultiple(of: 32)
                        else { return nil }
                        return (groupSize: 32, bits: 4, mode: .affine)
                    }
                }
            }
            var checkpoint = [String: MLXArray]()
            for (key, value) in original.parameters().flattened() {
                if key.contains(".switch_mlp.") {
                    for expert in 0 ..< value.dim(0) {
                        checkpoint[
                            key.replacingOccurrences(
                                of: ".switch_mlp.", with: ".experts.\(expert).")] = value[expert]
                    }
                } else if key.contains(".gate.e_score_correction_bias") {
                    checkpoint[
                        key.replacingOccurrences(
                            of: ".gate.e_score_correction_bias", with: ".e_score_correction_bias")] =
                        value
                } else if name == "mamba" {
                    checkpoint[String(key.dropFirst("language_model.".count))] =
                        key.hasSuffix("conv1d.weight")
                        ? value.swappedAxes(1, 2) : value
                } else {
                    checkpoint[key] = value
                }
            }
            if name == "exaone_moe" { checkpoint["mtp.unused.weight"] = MLXArray.zeros([1]) }
            try restored.update(
                parameters: ModuleParameters.unflattened(restored.sanitize(weights: checkpoint)),
                verify: .all)
            let tokens = MLXArray([1, 7, 4, 2, 3], [1, 5])
            let expected = original(tokens, cache: try original.newCache(configuration: nil))
            let actual = restored(tokens, cache: try restored.newCache(configuration: nil))
            XCTAssertEqual(
                actual.asArray(Float.self), expected.asArray(Float.self),
                "\(name), compressed: \(compressed)")
        }
    }
}
