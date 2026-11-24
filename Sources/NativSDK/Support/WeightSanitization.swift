// Copyright © 2024 Apple Inc.

import MLX

/// Native text models that also accept Python's language-model wrapper namespace.
protocol UnwrappedLanguageModelWeights {
    func checkpointQuantization(_ original: BaseConfiguration.PerLayerQuantization?) throws
        -> BaseConfiguration.PerLayerQuantization?
}

func resolveLanguageModelQuantization(
    for model: any LanguageModel, metadata: BaseConfiguration.PerLayerQuantization?,
    sidecar: BaseConfiguration.Quantization?
) throws -> BaseConfiguration.PerLayerQuantization? {
    guard let model = model as? any UnwrappedLanguageModelWeights else { return metadata }
    return try model.checkpointQuantization(
        metadata ?? sidecar.map { .init(quantization: $0, perLayerQuantization: [:]) })
}

extension UnwrappedLanguageModelWeights {
    static func unwrapLanguageModelWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights
        for (name, weight) in weights where name.hasPrefix("language_model.") {
            let path = String(name.dropFirst("language_model.".count))
            // Leave ambiguous duplicates in place for strict checkpoint validation.
            guard result[path] == nil else { continue }
            result[path] = weight
            result.removeValue(forKey: name)
        }
        return result
    }

    func checkpointQuantization(_ original: BaseConfiguration.PerLayerQuantization?) throws
        -> BaseConfiguration.PerLayerQuantization?
    {
        try Self.unwrapLanguageModelQuantization(original)
    }

    static func unwrapLanguageModelQuantization(_ original: BaseConfiguration.PerLayerQuantization?)
        throws -> BaseConfiguration.PerLayerQuantization?
    {
        guard var original else { return nil }
        for (name, option) in original.perLayerQuantization where name.hasPrefix("language_model.")
        {
            let path = String(name.dropFirst("language_model.".count))
            guard original.perLayerQuantization[path] == nil else {
                throw ModelExecutionError.invalidOptions("Duplicate quantization path: \(path)")
            }
            original.perLayerQuantization[path] = option
            original.perLayerQuantization.removeValue(forKey: name)
        }
        return original
    }
}

/// Removes checkpoint tensors owned by an `lm_head` module when the model uses its token
/// embedding as the output projection instead.
///
/// Quantized linear layers carry parameters in addition to `weight` (for example `scales`
/// and `biases`). Filtering by the module path keeps those parameters from being loaded into
/// the absent head. Matching a complete path component also supports weights that have already
/// been namespaced by a wrapper model without affecting similarly named modules.
package func filterLMHeadWeights(
    from weights: [String: MLXArray], tiedWordEmbeddings: Bool
) -> [String: MLXArray] {
    guard tiedWordEmbeddings else { return weights }

    return weights.filter { key, _ in
        !key.split(separator: ".").contains("lm_head")
    }
}
