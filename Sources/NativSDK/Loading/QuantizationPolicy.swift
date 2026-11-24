import MLX
import MLXNN

/// A model's decision for one quantizable module, before conversion checks its shape.
public enum QuantizationDecision: Sendable {
    case skip
    case useDefault
    case parameters(BaseConfiguration.Quantization)
}

/// The default conversion policy owned by an architecture. Each layer can be skipped,
/// use the requested defaults, or specify its own quantization parameters.
/// An explicit recipe or custom predicate replaces this policy during conversion.
public protocol QuantizationPolicyProviding {
    func quantizationPredicate(path: String, module: Module) -> QuantizationDecision
}

/// An optional architecture-owned mixed format, selected explicitly during conversion.
/// Explicit parameters can opt vision/audio modules into a model's mixed recipe.
public protocol MixedQuantizationPolicyProviding {
    func mixedQuantizationPredicate(path: String, module: Module) -> QuantizationDecision
}

extension QuantizationPolicyProviding {
    public func quantizationPredicate(path: String, module: Module) -> QuantizationDecision {
        .useDefault
    }
}

extension BaseConfiguration.Quantization {
    func validateForConversion() throws {
        let valid: Bool
        switch mode {
        case .affine:
            valid = [32, 64, 128].contains(groupSize) && [2, 3, 4, 5, 6, 8].contains(bits)
        case .mxfp4: valid = groupSize == 32 && bits == 4
        case .mxfp8: valid = groupSize == 32 && bits == 8
        case .nvfp4: valid = groupSize == 16 && bits == 4
        }
        guard valid else {
            throw ModelExecutionError.invalidOptions(
                "convert: unsupported quantization: \(mode.rawValue), group size \(groupSize), \(bits) bits"
            )
        }
    }

    var jsonObject: [String: Any] {
        ["group_size": groupSize, "bits": bits, "mode": mode.rawValue]
    }
}
