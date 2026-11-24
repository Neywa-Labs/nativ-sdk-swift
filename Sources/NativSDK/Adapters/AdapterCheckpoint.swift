import Foundation
import MLX
import MLXNN

public enum ModelAdapterError: LocalizedError {
    case unsupportedModel
    case alreadyLoaded
    case notLoaded
    case invalidCheckpoint(String)
    case restorationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedModel: "This model does not expose adapter-compatible layers."
        case .alreadyLoaded: "Unload or fuse the active adapter before loading another."
        case .notLoaded: "No adapter is loaded."
        case .invalidCheckpoint(let reason): "Invalid adapter checkpoint: \(reason)"
        case .restorationFailed(let reason):
            "Unable to restore the model after an adapter update: \(reason)"
        }
    }
}

/// Decoded and used only while the model owns exclusive model access.
struct AdapterCheckpoint {
    enum Kind: String, Decodable { case lora, dora }
    let kind: Kind
    let rank: Int
    let scale: Float
    let dropout: Float
    let numLayers: Int
    let keys: [String]?
    let peft: Bool
    let weights: [String: MLXArray]

    init(directory: URL) throws {
        let data = try Data(contentsOf: directory.appendingPathComponent("adapter_config.json"))
        let object = try JSONDecoder.json5().decode([String: JSONValue].self, from: data)
        peft = object["peft_type"] != nil
        if peft {
            struct Configuration: Decodable {
                let peft_type: String
                let r: Int
                let lora_alpha: Float
                let lora_dropout: Float?
                let target_modules: [String]
                let use_dora: Bool?
                let use_rslora: Bool?
                let bias: String?
            }
            let config = try JSONDecoder.json5().decode(Configuration.self, from: data)
            guard config.peft_type.uppercased() == "LORA", config.use_dora != true,
                config.use_rslora != true, config.bias == nil || config.bias == "none",
                config.r > 0, !config.target_modules.isEmpty
            else {
                throw ModelAdapterError.invalidCheckpoint(
                    "Only standard PEFT LoRA without bias updates is supported")
            }
            // These variants change scaling, parameter selection, or the base weights.
            for key in [
                "rank_pattern", "alpha_pattern", "modules_to_save", "layers_to_transform",
                "layer_replication", "megatron_config", "target_parameters",
            ] {
                if let value = object[key], value != .null, value != .object([:]),
                    value != .array([])
                {
                    throw ModelAdapterError.invalidCheckpoint("Unsupported PEFT option: \(key)")
                }
            }
            kind = .lora
            rank = config.r
            scale = config.lora_alpha / Float(config.r)
            dropout = config.lora_dropout ?? 0
            numLayers = Int.max
            keys = config.target_modules
            let raw = try MLX.loadArrays(
                url: directory.appendingPathComponent("adapter_model.safetensors"))
            var converted: [String: MLXArray] = [:]
            for (name, tensor) in raw {
                var name = name
                if name.hasPrefix("base_model.model.") {
                    name = String(name.dropFirst("base_model.model.".count))
                }
                let suffix: String
                let parameter: String
                if name.hasSuffix(".lora_A.weight") {
                    suffix = ".lora_A.weight"
                    parameter = ".lora_a"
                } else if name.hasSuffix(".lora_B.weight") {
                    suffix = ".lora_B.weight"
                    parameter = ".lora_b"
                } else {
                    throw ModelAdapterError.invalidCheckpoint("Unsupported PEFT tensor: \(name)")
                }
                guard tensor.ndim == 2 else {
                    throw ModelAdapterError.invalidCheckpoint("Expected a matrix: \(name)")
                }
                let target = String(name.dropLast(suffix.count)) + parameter
                guard converted[target] == nil else {
                    throw ModelAdapterError.invalidCheckpoint("Duplicate PEFT tensor: \(target)")
                }
                converted[target] = tensor.transposed()
            }
            weights = converted
        } else {
            struct Configuration: Decodable {
                let fine_tune_type: Kind
                let num_layers: Int
                let lora_parameters: Parameters
                struct Parameters: Decodable {
                    let rank: Int
                    let scale: Float
                    let dropout: Float?
                    let keys: [String]?
                }
            }
            let config = try JSONDecoder.json5().decode(Configuration.self, from: data)
            kind = config.fine_tune_type
            rank = config.lora_parameters.rank
            scale = config.lora_parameters.scale
            dropout = config.lora_parameters.dropout ?? 0
            numLayers = config.num_layers
            keys = config.lora_parameters.keys
            weights = try MLX.loadArrays(
                url: directory.appendingPathComponent("adapters.safetensors"))
        }
        guard rank > 0, scale.isFinite, (0 ..< 1).contains(dropout), numLayers > 0,
            !weights.isEmpty, keys?.isEmpty != true
        else {
            throw ModelAdapterError.invalidCheckpoint(
                "Invalid rank, scale, dropout, layer count or empty weights")
        }
    }

    func prepare(for model: any LanguageModel) throws -> InstalledAdapter {
        guard let lora = model as? any LoRAModel else { throw ModelAdapterError.unsupportedModel }
        let named = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        var eligible: [String: Linear] = [:]
        let selectedKeys = keys ?? lora.loraDefaultKeys
        for layer in lora.loraLayers.suffix(numLayers) {
            guard let root = named.first(where: { $0.value === layer })?.key else {
                throw ModelAdapterError.invalidCheckpoint(
                    "Adapter layer is absent from the model tree")
            }
            for (relative, module) in layer.namedModules() {
                let path = root.isEmpty ? relative : root + "." + relative
                let selected =
                    peft
                    ? selectedKeys.contains(where: {
                        relative == $0 || path == $0 || relative.hasSuffix("." + $0)
                    })
                    : selectedKeys.contains(relative)
                guard selected, let linear = module as? Linear else { continue }
                guard !(linear is any LoRALayer),
                    type(of: linear) == Linear.self || type(of: linear) == QuantizedLinear.self
                else {
                    throw ModelAdapterError.invalidCheckpoint(
                        "Unsupported or already adapted projection: \(path)")
                }
                eligible[path] = linear
            }
        }
        guard !eligible.isEmpty else {
            throw ModelAdapterError.invalidCheckpoint("No matching linear layers")
        }
        var grouped: [String: [String: MLXArray]] = [:]
        for (name, tensor) in weights {
            guard let dot = name.lastIndex(of: ".") else {
                throw ModelAdapterError.invalidCheckpoint("Invalid tensor name: \(name)")
            }
            var path = String(name[..<dot])
            let parameter = String(name[name.index(after: dot)...])
            guard ["lora_a", "lora_b", "m"].contains(parameter), tensor.dtype.isFloatingPoint else {
                throw ModelAdapterError.invalidCheckpoint("Unexpected adapter tensor: \(name)")
            }
            if eligible[path] == nil, peft {
                // Vision wrappers may add a language_model prefix around the text backbone.
                let matches = eligible.keys.filter { $0.hasSuffix("." + path) }
                if matches.count == 1 { path = matches[0] }
            }
            guard eligible[path] != nil, grouped[path]?[parameter] == nil else {
                throw ModelAdapterError.invalidCheckpoint("Unknown or duplicate tensor: \(name)")
            }
            grouped[path, default: [:]][parameter] = tensor
        }
        // Native checkpoints describe every selected layer; PEFT tensors select specific layers.
        guard peft || Set(grouped.keys) == Set(eligible.keys) else {
            throw ModelAdapterError.invalidCheckpoint(
                "Missing weights for configured adapter layers")
        }
        var originals: [String: Module] = [:]
        var adapted: [String: Module] = [:]
        for (path, tensors) in grouped {
            let base = eligible[path]!
            let (output, input) = base.shape
            let expected =
                kind == .dora ? Set(["lora_a", "lora_b", "m"]) : Set(["lora_a", "lora_b"])
            guard Set(tensors.keys) == expected,
                tensors["lora_a"]?.shape == [input, rank],
                tensors["lora_b"]?.shape == [rank, output],
                kind != .dora || tensors["m"]?.shape == [output]
            else {
                throw ModelAdapterError.invalidCheckpoint(
                    "Missing or incompatible weights at \(path)")
            }
            let replacement =
                kind == .lora
                ? LoRALinear.from(linear: base, rank: rank, scale: scale, dropout: dropout)
                : DoRALinear.from(linear: base, rank: rank, scale: scale, dropout: dropout)
            try replacement.update(
                parameters: .unflattened(tensors), verify: [.noUnusedKeys, .shapeMismatch])
            replacement.train(false)
            eval(replacement)
            originals[path] = base
            adapted[path] = replacement
        }
        return InstalledAdapter(originals: originals, adapted: adapted)
    }
}

struct InstalledAdapter {
    let originals: [String: Module]
    let adapted: [String: Module]
}
