import Foundation
import MLX

/// Encoder inference, independent of token generation and conversation caches.
public protocol EmbeddingModel: BaseLanguageModel {
    var maximumSequenceLength: Int { get }
    func embed(_ input: LMInput) throws -> EmbeddingOutput
}

/// Evaluated inside the model; tensors never cross the public operation boundary.
public enum EmbeddingOutput {
    /// Shape [batch, dimensions].
    case pooled(MLXArray)
    /// Shape [batch, tokens, dimensions]. Padding is removed when producing host results.
    case tokens(MLXArray)
}

/// Pooling is a checkpoint convention, not a sampling setting.
enum EmbeddingPooling: String, Sendable {
    case mean, cls, max, lasttoken

    static func load(from directory: URL, default fallback: Self) throws -> Self {
        let path = directory.appendingPathComponent("1_Pooling/config.json")
        guard FileManager.default.fileExists(atPath: path.path) else { return fallback }
        let object = try JSONDecoder.json5().decode(
            [String: JSONValue].self, from: Data(contentsOf: path))
        if case .bool(false) = object["include_prompt"] {
            throw ModelExecutionError.invalidOptions(
                "Prompt-excluding embedding pooling is unsupported")
        }
        if let value = object["pooling_mode"] {
            guard case .string(let name) = value, let mode = Self(rawValue: name) else {
                throw ModelExecutionError.invalidOptions("Unsupported embedding pooling mode")
            }
            return mode
        }
        let flags: [(String, Self?)] = [
            ("pooling_mode_cls_token", .cls), ("pooling_mode_max_tokens", .max),
            ("pooling_mode_mean_tokens", .mean), ("pooling_mode_lasttoken", .lasttoken),
            ("pooling_mode_mean_sqrt_len_tokens", nil), ("pooling_mode_weightedmean_tokens", nil),
        ]
        guard flags.contains(where: { object[$0.0] != nil }) else {
            return fallback
        }
        for (key, _) in flags where object[key] != nil {
            guard case .bool = object[key] else {
                throw ModelExecutionError.invalidOptions("Embedding pooling flags must be Boolean")
            }
        }
        let enabled = flags.filter { object[$0.0] == .bool(true) }
        guard enabled.count <= 1, enabled.isEmpty || enabled[0].1 != nil else {
            throw ModelExecutionError.invalidOptions(
                "Unsupported or concatenated embedding pooling modes")
        }
        return enabled.first?.1 ?? .mean
    }

    func apply(_ hidden: MLXArray, mask: MLXArray) -> MLXArray {
        let valid = mask.asType(.bool)
        switch self {
        case .mean:
            return sum(
                which(
                    valid.expandedDimensions(axis: -1), hidden.asType(.float32), MLXArray(Float(0))),
                axis: 1)
                / maximum(sum(valid.asType(.float32), axis: 1, keepDims: true), 1)
        case .max:
            return MLX.max(
                which(valid.expandedDimensions(axis: -1), hidden, MLXArray(-Float.infinity)),
                axis: 1)
        case .cls, .lasttoken:
            let index =
                self == .cls
                ? argMax(valid.asType(.int32), axis: 1)
                : (mask.dim(1) - 1 - argMax(valid[0..., .stride(by: -1)].asType(.int32), axis: 1))
            return takeAlong(hidden, index.reshaped([-1, 1, 1]), axis: 1).squeezed(axis: 1)
        }
    }
}

func normalizeEmbeddings(_ values: MLXArray) -> MLXArray {
    let values = values.asType(.float32)
    return values / maximum(sqrt(sum(square(values), axis: -1, keepDims: true)), 1e-9)
}

/// Validate tokens once before dispatching to an encoder's architecture-specific layers.
func embeddingTokens(
    _ input: LMInput, vocabularySize: Int, maximumLength: Int,
    allowsMedia: Bool = false
) throws -> (MLXArray, MLXArray) {
    guard input.audio == nil, allowsMedia || (input.image == nil && input.video == nil) else {
        throw ModelExecutionError.invalidOptions("Unsupported media for this embedding model")
    }
    var ids = input.text.tokens
    if ids.ndim == 1 { ids = ids.expandedDimensions(axis: 0) }
    guard ids.ndim == 2, ids.dim(0) > 0, ids.dim(1) > 0, ids.dim(1) <= maximumLength,
        ids.dtype == .int32 || ids.dtype == .int64,
        ids.min().item(Int.self) >= 0, ids.max().item(Int.self) < vocabularySize
    else {
        throw ModelExecutionError.invalidOptions("Invalid embedding token IDs or sequence length")
    }
    var mask = input.text.mask ?? MLXArray.ones(ids.shape, dtype: .bool)
    if mask.ndim == 1 { mask = mask.expandedDimensions(axis: 0) }
    guard mask.shape == ids.shape, all((mask .== 0) .|| (mask .== 1)).item(Bool.self),
        all(sum(mask.asType(.int32), axis: 1) .> 0).item(Bool.self)
    else {
        throw ModelExecutionError.invalidOptions(
            "Embedding masks require a nonempty binary sequence per input")
    }
    return (ids, mask.asType(.bool))
}

func embeddingAttentionMask(_ valid: MLXArray, causal: Bool = false) -> MLXArray {
    let keys = valid.expandedDimensions(axes: [1, 2])
    guard causal else { return keys }
    let positions = arange(valid.dim(1))
    return keys
        .&& (positions.expandedDimensions(axis: 1) .>= positions)
        .reshaped([1, 1, valid.dim(1), valid.dim(1)])
}

func embeddingBackboneWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var result: [String: MLXArray] = [:]
    for (name, value) in weights where !name.hasPrefix("lm_head.") {
        result[name.hasPrefix("model.") ? name : "model." + name] = value
    }
    return result
}
