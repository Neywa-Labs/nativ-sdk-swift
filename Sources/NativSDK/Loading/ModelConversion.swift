import Foundation
import MLX
import MLXNN

/// Quantizes a local MLX checkpoint using its model policy or an explicit mixed recipe,
/// preserving already packed weights.
/// Run offline on a Mac. The source must already use the model's MLX tensor layout.
public enum ModelConversion {
    public typealias Predicate = (String, Module) -> QuantizationDecision

    public struct Result: Sendable {
        public let shardCount: Int
        public let tensorBytes: Int
        public let quantizedMatrixCount: Int
    }

    /// Writes a new checkpoint without modifying the source or overwriting a destination.
    /// The shard size is a target: an individual matrix and its scales may exceed it.
    /// The destination becomes visible only after all weights and sidecars are written.
    public static func convert(
        source: URL, destination: URL,
        quantization requested: BaseConfiguration.Quantization = .init(groupSize: 64, bits: 4),
        recipe: QuantizationRecipe? = nil,
        predicate: Predicate? = nil,
        shardSizeBytes: Int = 512 * 1024 * 1024,
        progress: (String) -> Void = { _ in }
    ) throws -> Result {
        guard source.isFileURL, destination.isFileURL else {
            throw invalid("source and destination must be local directories")
        }
        guard recipe == nil || predicate == nil else {
            throw invalid("choose either a recipe or a custom predicate")
        }
        try requested.validateForConversion()
        let defaults = recipe?.defaults ?? requested
        guard shardSizeBytes > 0 else { throw invalid("shard size must be positive") }
        let manager = FileManager.default
        let source = source.standardizedFileURL
        let destination = destination.standardizedFileURL
        guard !manager.fileExists(atPath: destination.path) else {
            throw invalid("destination already exists: \(destination.path)")
        }
        var config = try readJSON(source.appendingPathComponent("config.json"))
        let configData = try JSONSerialization.data(withJSONObject: config)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
        var quantization: [String: Any]
        if let existing = config["quantization"], !(existing is NSNull) {
            guard let values = existing as? [String: Any] else {
                throw invalid("invalid quantization metadata")
            }
            quantization = values
        } else {
            quantization = defaults.jsonObject
        }

        let indexURL = source.appendingPathComponent("model.safetensors.index.json")
        var index: [String: String]?
        let files: [String]
        if manager.fileExists(atPath: indexURL.path) {
            guard let map = try readJSON(indexURL)["weight_map"] as? [String: String],
                !map.isEmpty
            else { throw invalid("invalid or empty safetensors index") }
            index = map
            files = Set(map.values).sorted()
        } else {
            files = try manager.contentsOfDirectory(atPath: source.path).filter {
                $0.hasPrefix("model") && $0.hasSuffix(".safetensors")
            }.sorted()
        }
        guard !files.isEmpty else { throw invalid("no model safetensors found") }
        var names = Set<String>()
        var tensorMetadata: [String: String]?
        for file in files {
            guard !file.hasPrefix("/"), !file.split(separator: "/").contains(".."),
                file.hasSuffix(".safetensors")
            else { throw invalid("invalid shard path: \(file)") }
            // MLX loads headers lazily; inspecting keys does not materialize the weights.
            let (arrays, metadata) = try loadArraysAndMetadata(
                url: source.appendingPathComponent(file))
            guard metadata["format"] == "mlx" else {
                throw invalid(
                    "source must be an MLX checkpoint; Transformers tensor layouts are not supported"
                )
            }
            if let tensorMetadata, tensorMetadata != metadata {
                throw invalid("inconsistent safetensor metadata")
            }
            tensorMetadata = metadata
            for name in arrays.keys {
                guard names.insert(name).inserted else {
                    throw invalid("duplicate tensor: \(name)")
                }
                if let index, index[name] != file {
                    throw invalid("index does not match tensor \(name) in \(file)")
                }
            }
        }
        if let index, names != Set(index.keys) {
            throw invalid("index references missing tensors")
        }
        for name in names where name.hasSuffix(".scales") {
            let path = String(name.dropLast(7))
            guard names.contains(path + ".weight") else { continue }
            guard let parameters = base.perLayerQuantization?.quantization(layer: path) else {
                throw invalid("packed weight has no quantization metadata: \(path)")
            }
            try parameters.validateForConversion()
        }
        // Model construction stays lazy. Retain only its decisions, so evaluating output
        // shards cannot materialize or retain the model's random initial parameters.
        let decisions = try conversionDecisions(
            configuration: configData, modelType: base.modelType, names: names,
            defaults: defaults, recipe: recipe, predicate: predicate)

        let parent = destination.deletingLastPathComponent()
        try manager.createDirectory(at: parent, withIntermediateDirectories: true)
        let staging = parent.appendingPathComponent(".convert-\(UUID().uuidString)")
        try manager.createDirectory(at: staging, withIntermediateDirectories: false)
        defer { try? manager.removeItem(at: staging) }
        var weights: [String: MLXArray] = [:]
        var weightMap: [String: String] = [:]
        var pendingBytes = 0
        var totalBytes = 0
        var shardCount = 0
        var quantizedCount = 0

        func flush() throws {
            guard !weights.isEmpty else { return }
            shardCount += 1
            let filename = String(format: "model-%05d.safetensors", shardCount)
            try save(
                arrays: weights, metadata: tensorMetadata ?? ["format": "mlx"],
                url: staging.appendingPathComponent(filename))
            for name in weights.keys { weightMap[name] = filename }
            totalBytes += pendingBytes
            weights.removeAll()
            pendingBytes = 0
            Memory.clearCache()
        }

        for file in files {
            progress("Converting \(file)")
            var arrays = try loadArrays(url: source.appendingPathComponent(file))
            for name in arrays.keys.sorted() {
                let value = arrays.removeValue(forKey: name)!
                let prefix = name.hasSuffix(".weight") ? String(name.dropLast(7)) : name
                let replacements: [String: MLXArray]
                let candidate =
                    name.hasSuffix(".weight") && value.ndim >= 2
                    && [DType.float32, .float16, .bfloat16].contains(value.dtype)
                    && !names.contains(prefix + ".scales") && value.size > 0
                if candidate, let parameters = decisions[prefix],
                    value.dim(-1) % parameters.groupSize == 0
                {
                    guard !names.contains(prefix + ".biases") else {
                        throw invalid("quantization biases without scales for \(prefix)")
                    }
                    let packed = quantized(
                        value, groupSize: parameters.groupSize, bits: parameters.bits,
                        mode: parameters.mode)
                    var converted = [name: packed.wq, prefix + ".scales": packed.scales]
                    if let biases = packed.biases { converted[prefix + ".biases"] = biases }
                    replacements = converted
                    quantization[prefix] = parameters.jsonObject
                    quantizedCount += 1
                } else {
                    replacements = [name: value]
                    if candidate { quantization[prefix] = false }
                }
                let bytes = replacements.values.reduce(0) { $0 + $1.nbytes }
                if pendingBytes > 0 && bytes > shardSizeBytes - pendingBytes { try flush() }
                eval(Array(replacements.values))
                weights.merge(replacements) { _, new in new }
                pendingBytes += bytes
                if pendingBytes >= shardSizeBytes { try flush() }
            }
        }
        try flush()
        for name in [
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "chat_template.jinja", "processor_config.json", "preprocessor_config.json",
            "generation_config.json", "tokenizer.model", "vocab.json", "merges.txt",
            "added_tokens.json",
        ] {
            let file = source.appendingPathComponent(name)
            if manager.fileExists(atPath: file.path) {
                // Hugging Face snapshots symlink sidecars to their blob cache. Copy the
                // contents so the prepared checkpoint also works outside that cache.
                try manager.copyItem(
                    at: file.resolvingSymlinksInPath(), to: staging.appendingPathComponent(name))
            }
        }
        config["quantization"] = quantization
        if config["quantization_config"] != nil { config["quantization_config"] = quantization }
        try writeJSON(config, to: staging.appendingPathComponent("config.json"))
        try writeJSON(
            ["metadata": ["total_size": totalBytes], "weight_map": weightMap],
            to: staging.appendingPathComponent("model.safetensors.index.json"))
        try manager.moveItem(at: staging, to: destination)
        return Result(
            shardCount: shardCount, tensorBytes: totalBytes, quantizedMatrixCount: quantizedCount)
    }

    private static func conversionDecisions(
        configuration: Data, modelType: String, names: Set<String>,
        defaults: BaseConfiguration.Quantization, recipe: QuantizationRecipe?, predicate: Predicate?
    ) throws -> [String: BaseConfiguration.Quantization] {
        let model: any BaseLanguageModel
        if MiniCPMOCRJevConfiguration.matches(configuration) {
            model = try MiniCPMOCRJev(
                JSONDecoder.json5().decode(MiniCPMOCRJevConfiguration.self, from: configuration))
        } else {
            model = try ModelRegistry.default.createModel(
                configuration: configuration, modelType: modelType, weightNames: names)
        }
        let selected = try predicate ?? recipe?.predicate(for: model)
        var result: [String: BaseConfiguration.Quantization] = [:]
        for (path, module) in model.namedModules()
        where module is Quantizable && !(module is Quantized) {
            let sourcePath =
                model is any UnwrappedLanguageModelWeights
                    && names.contains("language_model." + path + ".weight")
                ? "language_model." + path : path
            guard names.contains(sourcePath + ".weight"), !names.contains(sourcePath + ".scales")
            else {
                continue
            }
            let decision: QuantizationDecision
            if let selected {
                decision = selected(path, module)
            } else {
                // Match Python's default wrapper: exclude multimodal modules, then
                // honor both model exclusions and model-owned per-layer parameters.
                if isMultimodal(path) { continue }
                decision = model.quantizationPredicate(path: path, module: module)
            }
            let parameters: BaseConfiguration.Quantization
            switch decision {
            case .skip: continue
            case .useDefault:
                if isMultimodal(path) { continue }
                parameters = defaults
            case .parameters(let specified): parameters = specified
            }
            try parameters.validateForConversion()
            result[sourcePath] = parameters
        }
        return result
    }

    static func isMultimodal(_ path: String) -> Bool {
        [
            "vision", "visual", "aligner", "vl_connector", "sam_model", "audio",
            "code_predictor", "img_projector", "multi_modal_projector", "patch_merge_mlp",
        ].contains(where: path.contains)
    }

    private static func invalid(_ message: String) -> ModelExecutionError {
        .invalidOptions("convert: " + message)
    }

    private static func readJSON(_ url: URL) throws -> [String: Any] {
        guard
            let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [String: Any]
        else { throw invalid("expected a JSON object in \(url.lastPathComponent)") }
        return object
    }

    private static func writeJSON(_ object: [String: Any], to url: URL) throws {
        var data = try JSONSerialization.data(
            withJSONObject: object, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        data.append(0x0A)
        try data.write(to: url)
    }
}
