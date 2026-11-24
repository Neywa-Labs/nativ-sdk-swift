import Foundation
import MLX
import MLXNN

/// compressed-tensors stores FP4 nibbles, E4M3 block scales, and an inverse global
/// scale. Keep all three: folding the global factor into E4M3 would round it again.
enum CompressedTensorsNVFP4 {
    struct Configuration: Decodable, Sendable {
        let quantMethod: String?
        let format: String?
        let configGroups: [String: Group]?

        enum CodingKeys: String, CodingKey {
            case quantMethod = "quant_method", format, configGroups = "config_groups"
        }

        struct Group: Decodable, Sendable {
            let weights: Weights
            let hasActivationQuantization: Bool

            enum CodingKeys: String, CodingKey {
                case weights, input = "input_activations", output = "output_activations"
            }

            init(from decoder: any Decoder) throws {
                let c = try decoder.container(keyedBy: CodingKeys.self)
                weights = try c.decode(Weights.self, forKey: .weights)
                hasActivationQuantization = try [CodingKeys.input, .output].contains {
                    try c.contains($0) && !c.decodeNil(forKey: $0)
                }
            }

        }

        struct Weights: Decodable, Sendable {
            let bits: Int
            let type: String
            let symmetric: Bool
            let groupSize: Int
            let strategy: String
            let dynamic: Bool
            let scaleDType: String

            enum CodingKeys: String, CodingKey {
                case bits = "num_bits", type, symmetric, groupSize = "group_size"
                case strategy, dynamic, scaleDType = "scale_dtype"
            }
        }

        func validate() throws {
            guard quantMethod == "compressed-tensors" else { return }
            guard format == "nvfp4-pack-quantized", let groups = configGroups, !groups.isEmpty,
                groups.values.allSatisfy({ group in
                    let w = group.weights
                    return !group.hasActivationQuantization && w.bits == 4 && w.type == "float"
                        && w.symmetric && w.groupSize == 16 && w.strategy == "tensor_group"
                        && !w.dynamic && w.scaleDType == "torch.float8_e4m3fn"
                })
            else {
                throw invalid("only static W4A16 nvfp4-pack-quantized checkpoints are supported")
            }
        }
    }

    static let quantization = BaseConfiguration.Quantization(groupSize: 16, bits: 4, mode: .nvfp4)

    static func invalid(_ message: String) -> ModelExecutionError {
        .invalidOptions("compressed-tensors NVFP4: " + message)
    }

    static func prepare(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var result = weights
        for (key, packed) in weights where key.hasSuffix(".weight_packed") {
            let path = String(key.dropLast(".weight_packed".count))
            guard packed.ndim == 2, packed.dtype == .uint8, packed.dim(0) > 0,
                packed.dim(1) > 0, packed.dim(1).isMultiple(of: 8),
                let scales = weights[path + ".weight_scale"], scales.dtype == .uint8,
                scales.shape == [packed.dim(0), packed.dim(1) / 8],
                let global = weights[path + ".weight_global_scale"], global.size == 1,
                [.float32, .float16, .bfloat16].contains(global.dtype),
                weights[path + ".weight"] == nil, weights[path + ".scales"] == nil
            else { throw invalid("missing or invalid packed matrix/scales for \(path)") }
            let scalar = global.asType(.float32, stream: .cpu).item(Float.self)
            guard scalar.isFinite, scalar > 0 else {
                throw invalid("invalid global scale for \(path)")
            }
            result.removeValue(forKey: key)
            result.removeValue(forKey: path + ".weight_scale")
            result[path + ".weight"] = packed.view(dtype: .uint32)
            result[path + ".scales"] = scales
            result[path + ".weight_global_scale"] = MLXArray(scalar)
        }
        guard !result.keys.contains(where: { $0.hasSuffix(".weight_scale") }) else {
            throw invalid("orphan block scales without packed weights")
        }
        return result
    }

    /// Install from existing arrays, avoiding evaluation of randomly initialized model weights.
    static func install(in model: Module, weights: [String: MLXArray]) throws {
        var replacements: [String: Module] = [:]
        var consumed = Set<String>()
        for (path, module) in model.namedModules() {
            let inverse = weights[path + ".weight_global_scale"] != nil
            let key = path + (inverse ? ".weight_global_scale" : ".weight_scale_2")
            guard let global = weights[key] else { continue }
            guard !(inverse && weights[path + ".weight_scale_2"] != nil) else {
                throw invalid("conflicting global scale conventions for \(path)")
            }
            guard let weight = weights[path + ".weight"], let scales = weights[path + ".scales"],
                weight.dtype == .uint32, scales.dtype == .uint8
            else { throw invalid("global scale without an NVFP4 matrix: \(path)") }
            if let projection = module as? SwitchLinear {
                guard global.shape == [projection.numExperts],
                    weight.shape == [
                        projection.numExperts, projection.outputDims, projection.inputDims / 8,
                    ],
                    scales.shape == [
                        projection.numExperts, projection.outputDims, projection.inputDims / 16,
                    ]
                else { throw invalid("invalid expert dimensions for \(path)") }
                replacements[path] = ScaledNVFP4SwitchLinear(
                    projection, weight: weight, scales: scales, global: global, inverse: inverse)
            } else if let linear = module as? Linear {
                guard global.size == 1, weight.shape == [linear.shape.0, linear.shape.1 / 8],
                    scales.shape == [linear.shape.0, linear.shape.1 / 16]
                else { throw invalid("invalid linear dimensions for \(path)") }
                replacements[path] = ScaledNVFP4Linear(
                    weight: weight, scales: scales, global: global, bias: linear.bias,
                    inverse: inverse)
            } else {
                throw invalid("global scale on an unsupported module: \(path)")
            }
            consumed.insert(key)
        }
        guard
            Set(
                weights.keys.filter {
                    $0.hasSuffix(".weight_global_scale") || $0.hasSuffix(".weight_scale_2")
                }) == consumed
        else {
            throw invalid("global scale has no matching model projection")
        }
        try replaceByParent(model: model, replacements: replacements)
    }
}

private final class ScaledNVFP4Linear: QuantizedLinear {
    @ParameterInfo(key: "weight_global_scale") var inverseGlobalScale: MLXArray?
    @ParameterInfo(key: "weight_scale_2") var multiplicativeGlobalScale: MLXArray?

    init(weight: MLXArray, scales: MLXArray, global: MLXArray, bias: MLXArray?, inverse: Bool) {
        _inverseGlobalScale.wrappedValue = inverse ? global : nil
        _multiplicativeGlobalScale.wrappedValue = inverse ? nil : global
        super.init(
            weight: weight, bias: bias, scales: scales, biases: nil,
            groupSize: 16, bits: 4, mode: .nvfp4)
        freeze()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let raw = quantizedMM(
            x.asType(.float32), weight, scales: scales, biases: nil, transpose: true,
            groupSize: 16, bits: 4, mode: .nvfp4)
        let result = (inverseGlobalScale.map { raw / $0 } ?? (raw * multiplicativeGlobalScale!))
            .asType(x.dtype)
        return bias.map { result + $0 } ?? result
    }
}

private final class ScaledNVFP4SwitchLinear: SwitchLinear, Quantized {
    let groupSize = 16
    let bits = 4
    let mode = QuantizationMode.nvfp4

    @ParameterInfo var scales: MLXArray
    @ParameterInfo(key: "weight_global_scale") var inverseGlobalScale: MLXArray?
    @ParameterInfo(key: "weight_scale_2") var multiplicativeGlobalScale: MLXArray?

    init(
        _ source: SwitchLinear, weight: MLXArray, scales: MLXArray, global: MLXArray, inverse: Bool
    ) {
        _scales.wrappedValue = scales
        _inverseGlobalScale.wrappedValue = inverse ? global : nil
        _multiplicativeGlobalScale.wrappedValue = inverse ? nil : global
        super.init(
            inputDims: source.inputDims, outputDims: source.outputDims,
            numExperts: source.numExperts, weight: weight, bias: source.bias)
        freeze()
    }

    override func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let raw = gatherQuantizedMM(
            x.asType(.float32), weight, scales: scales, biases: nil, rhsIndices: indices,
            transpose: true, groupSize: 16, bits: 4, mode: .nvfp4, sortedIndices: sortedIndices)
        let global = (inverseGlobalScale ?? multiplicativeGlobalScale!)[indices].expandedDimensions(
            axes: [
                -1, -2,
            ])
        let result = (inverseGlobalScale != nil ? raw / global : raw * global).asType(x.dtype)
        return bias.map { result + $0[indices].expandedDimensions(axis: -2) } ?? result
    }
}
