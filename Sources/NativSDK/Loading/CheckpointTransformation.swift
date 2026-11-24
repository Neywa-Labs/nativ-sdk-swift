import Foundation
import MLX
import MLXNN

/// Resident loading and expert streaming share checkpoint-format handling.
/// Models sanitize architecture-specific names and layouts after format normalization.
enum CheckpointTransformation {
    case none
    case compressedTensorsNVFP4
    case modelOpt

    init(modelDirectory: URL) throws {
        let url = modelDirectory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            self = .none
            return
        }
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
        guard let config = object as? [String: Any] else {
            throw ModelExecutionError.invalidOptions("Checkpoint config must be a JSON object")
        }
        // Native MLX quantization is authoritative. Converted checkpoints may
        // retain source metadata that no longer describes their tensor storage.
        // Inkling 2 publishes a ModelOpt-compatible label with its own packed
        // expert layout and weight transforms in hf_quant_config.json.
        if config["quantization"] is [String: Any] || config["model_type"] as? String == "inkling2"
        {
            self = .none
            return
        }
        guard let quantization = config["quantization_config"] as? [String: Any] else {
            self = .none
            return
        }
        if let method = quantization["quant_method"] as? String,
            ["modelopt", "modelopt_mixed"].contains(method)
        {
            // Spark's packed experts declare their geometry through the model and
            // tensor shapes. Some exports have no ModelOpt config_groups dictionary.
            let sparkExpertLayout =
                (config["model_type"] as? String).map(MuseSparkConfiguration.matches) == true
                && quantization["quant_algo"] as? String == "NVFP4"
                && quantization["config_groups"] == nil
            if !sparkExpertLayout { try ModelOptCheckpoint.validate(quantization) }
            self = .modelOpt
            return
        }
        guard quantization["quant_method"] as? String == "compressed-tensors" else {
            self = .none
            return
        }
        let source = try JSONDecoder().decode(
            CompressedTensorsNVFP4.Configuration.self,
            from: JSONSerialization.data(withJSONObject: quantization))
        try source.validate()
        self = .compressedTensorsNVFP4
    }

    /// Normalize storage representations before model-specific sanitization.
    func prepare(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        switch self {
        case .none: weights
        case .compressedTensorsNVFP4: try CompressedTensorsNVFP4.prepare(weights)
        case .modelOpt: try ModelOptCheckpoint.prepare(weights)
        }
    }

    /// Projection installation uses the final model paths produced by sanitization.
    func install(in model: Module, weights: [String: MLXArray]) throws {
        switch self {
        case .none:
            // Native mixed checkpoints can retain already packed NVFP4 projections.
            // Their global scales remain separate from the block scales.
            if weights.keys.contains(where: {
                $0.hasSuffix(".weight_global_scale") || $0.hasSuffix(".weight_scale_2")
            }) {
                try CompressedTensorsNVFP4.install(in: model, weights: weights)
            }
        case .compressedTensorsNVFP4, .modelOpt:
            try CompressedTensorsNVFP4.install(in: model, weights: weights)
        }
    }

    var expertQuantization: BaseConfiguration.Quantization? {
        switch self {
        case .none: nil
        case .compressedTensorsNVFP4, .modelOpt: CompressedTensorsNVFP4.quantization
        }
    }

    /// Expert streaming resolves source byte ranges before materializing expert arrays.
    func sourceTensorName(for path: String) -> String {
        if case .none = self { return path }
        if case .compressedTensorsNVFP4 = self, path.hasSuffix(".weight") {
            return String(path.dropLast(".weight".count)) + ".weight_packed"
        }
        if path.hasSuffix(".scales") {
            return String(path.dropLast(".scales".count)) + ".weight_scale"
        }
        return path
    }

    func prepareExpertTensor(
        _ tensor: ExpertWeightStore.Tensor, for path: String
    ) throws -> ExpertWeightStore.Tensor {
        if case .none = self { return tensor }
        // ModelOpt also uses a one-element vector for some scalar scale exports.
        if path.hasSuffix(".weight_scale_2"), tensor.shape == [1] {
            return .init(
                storage: tensor.storage, shape: [], dtype: tensor.dtype,
                bytesPerExpert: tensor.bytesPerExpert)
        }
        guard path.hasSuffix(".weight") else { return tensor }
        guard tensor.dtype == .uint8, tensor.shape.count == 2,
            tensor.shape[1].isMultiple(of: 8)
        else { throw CompressedTensorsNVFP4.invalid("invalid packed expert \(path)") }
        return .init(
            storage: tensor.storage, shape: [tensor.shape[0], tensor.shape[1] / 4],
            dtype: .uint32, bytesPerExpert: tensor.bytesPerExpert)
    }
}
