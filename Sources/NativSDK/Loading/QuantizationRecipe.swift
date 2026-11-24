import MLXNN

/// Explicit mixed formats. The named affine recipes follow mlx-vlm's layer schedule;
/// `mixed` selects the architecture's own format, when one is available.
public enum QuantizationRecipe: String, CaseIterable, Sendable {
    case mixed
    case mixed2_6 = "mixed_2_6"
    case mixed3_4 = "mixed_3_4"
    case mixed3_5 = "mixed_3_5"
    case mixed3_6 = "mixed_3_6"
    case mixed3_8 = "mixed_3_8"
    case mixed4_6 = "mixed_4_6"
    case mixed4_8 = "mixed_4_8"

    var bitWidths: (low: Int, high: Int)? {
        switch self {
        case .mixed: nil
        case .mixed2_6: (2, 6)
        case .mixed3_4: (3, 4)
        case .mixed3_5: (3, 5)
        case .mixed3_6: (3, 6)
        case .mixed3_8: (3, 8)
        case .mixed4_6: (4, 6)
        case .mixed4_8: (4, 8)
        }
    }

    var defaults: BaseConfiguration.Quantization? {
        bitWidths.map { .init(groupSize: 64, bits: $0.low) }
    }

    func predicate(for model: any BaseLanguageModel) throws -> ModelConversion.Predicate {
        guard let bits = bitWidths else {
            guard let policy = model as? any MixedQuantizationPolicyProviding else {
                throw ModelExecutionError.invalidOptions(
                    "convert: this model has no mixed recipe; use the default model policy or a named mixed-bit recipe"
                )
            }
            return { policy.mixedQuantizationPredicate(path: $0, module: $1) }
        }

        let paths = model.namedModules().map(\.0).filter { !ModelConversion.isMultimodal($0) }
        guard let down = paths.first(where: Self.isDownProjection),
            let layerLocation = down.split(separator: ".").firstIndex(where: { Int($0) != nil })
        else {
            throw ModelExecutionError.invalidOptions(
                "convert: \(rawValue) requires indexed decoder layers with down_proj or feed_forward.w2"
            )
        }
        let prefix = Array(down.split(separator: ".").prefix(layerLocation))
        let layerIndices = Set(paths.compactMap { path -> Int? in
            let parts = path.split(separator: ".")
            guard parts.count > layerLocation, Array(parts.prefix(layerLocation)) == prefix else {
                return nil
            }
            return Int(parts[layerLocation])
        })
        let count = layerIndices.count
        guard count > 0, layerIndices == Set(0 ..< count) else {
            throw ModelExecutionError.invalidOptions("convert: mixed recipe requires contiguous decoder layers")
        }
        return { path, _ in
            if ModelConversion.isMultimodal(path) { return .skip }
            let parts = path.split(separator: ".")
            let index = parts.count > layerLocation ? Int(parts[layerLocation]) ?? 0 : 0
            let useMoreBits = index < count / 8 || index >= 7 * count / 8
                || (index - count / 8) % 3 == 2
            let high = path.contains("lm_head") || path.contains("embed_tokens")
                || (useMoreBits && (path.contains("v_proj") || Self.isDownProjection(path)))
            return .parameters(.init(groupSize: 64, bits: high ? bits.high : bits.low))
        }
    }

    private static func isDownProjection(_ path: String) -> Bool {
        path.contains("down_proj") || path.contains("feed_forward.w2")
    }
}
