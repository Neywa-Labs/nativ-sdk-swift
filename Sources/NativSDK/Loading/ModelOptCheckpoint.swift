import Foundation
import MLX

/// ModelOpt exports packed NVFP4 matrices with multiplicative global scales and
/// per-tensor FP8 resident matrices. MLX executes these with floating activations;
/// NVIDIA's activation and KV-cache calibration scales are not model parameters.
enum ModelOptCheckpoint {
    static func invalid(_ message: String) -> ModelExecutionError {
        .invalidOptions("ModelOpt checkpoint: " + message)
    }

    static func validate(_ config: [String: Any]) throws {
        guard let algorithm = config["quant_algo"] as? String,
            ["NVFP4", "W4A16_NVFP4", "MIXED_PRECISION"].contains(algorithm),
            let groups = config["config_groups"] as? [String: [String: Any]], !groups.isEmpty,
            groups.values.allSatisfy({ group in
                guard let weights = group["weights"] as? [String: Any],
                    weights["type"] as? String == "float",
                    weights["dynamic"] as? Bool != true,
                    let bits = weights["num_bits"] as? Int
                else { return false }
                return bits == 8 || (bits == 4 && weights["group_size"] as? Int == 16)
            })
        else { throw invalid("only static NVFP4 and mixed NVFP4/FP8 weights are supported") }
        if let layers = config["quantized_layers"] as? [String: [String: Any]],
            !layers.values.allSatisfy({
                ["NVFP4", "FP8"].contains($0["quant_algo"] as? String ?? "")
            })
        {
            throw invalid("unsupported per-layer quantization algorithm")
        }
    }

    static func prepare(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var result = weights.filter { key, _ in
            ![".input_scale", ".k_scale", ".v_scale"].contains(where: key.hasSuffix)
        }
        for (key, scale) in weights where key.hasSuffix(".weight_scale") {
            let path = String(key.dropLast(".weight_scale".count))
            guard let weight = weights[path + ".weight"], weight.dtype == .uint8,
                weight.ndim == 2, weights[path + ".scales"] == nil
            else { throw invalid("missing or invalid matrix for \(path)") }
            if let global = weights[path + ".weight_scale_2"] {
                guard scale.dtype == .uint8, weight.dim(1).isMultiple(of: 8),
                    scale.shape == [weight.dim(0), weight.dim(1) / 8], global.size == 1,
                    [.float32, .float16, .bfloat16].contains(global.dtype)
                else { throw invalid("invalid NVFP4 scales for \(path)") }
                let scalar = global.asType(.float32, stream: .cpu).item(Float.self)
                guard scalar.isFinite, scalar > 0 else {
                    throw invalid("invalid NVFP4 global scale for \(path)")
                }
                result[path + ".weight"] = weight.view(dtype: .uint32)
                result[path + ".scales"] = scale
                result[path + ".weight_scale_2"] = MLXArray(scalar)
            } else {
                guard [.float32, .float16, .bfloat16].contains(scale.dtype),
                    scale.size == 1 || scale.shape == [weight.dim(0), 1]
                else { throw invalid("invalid FP8 scale for \(path)") }
                let decoded = decodeE4M3(weight)
                let matrix = (decoded * scale.asType(.float32)).asType(.bfloat16)
                // Materialize one resident matrix at a time, releasing its conversion graph.
                eval(matrix)
                result[path + ".weight"] = matrix
            }
            result.removeValue(forKey: key)
        }
        for key in result.keys where key.hasSuffix(".weight_scale_2") {
            let path = String(key.dropLast(".weight_scale_2".count))
            guard result[path + ".scales"] != nil else {
                throw invalid("global scale without packed weights for \(path)")
            }
        }
        return result
    }

    static func decodeE4M3(_ bytes: MLXArray) -> MLXArray {
        let values: [Float] = (0 ..< 256).map { byte in
            let exponent = (byte >> 3) & 15
            let mantissa = byte & 7
            let value: Float
            if exponent == 0 {
                value = Float(mantissa) / 512
            } else if exponent == 15 && mantissa == 7 {
                value = .nan
            } else {
                value = (1 + Float(mantissa) / 8) * pow(2, Float(exponent - 7))
            }
            return byte & 128 == 0 ? value : -value
        }
        return MLXArray(values)[bytes.asType(.int32)]
    }
}
