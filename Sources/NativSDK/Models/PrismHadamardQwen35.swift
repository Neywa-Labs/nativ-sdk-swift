import Foundation
import MLX
import MLXNN

/// Applies the pack's signed, normalized blockwise Walsh-Hadamard rotation.
///
/// The signs multiply before the transform on the way in and after it on the way out, so an
/// embedding lookup that rotates back is not the forward rotation run twice.
func hadamardRotate(
    _ x: MLXArray, block: Int, signs: MLXArray, inverse: Bool = false
) -> MLXArray {
    let shape = x.shape
    let dtype = x.dtype
    // The pack was calibrated with float32 rotations, including for the float16 lookup.
    var value = x.asType(.float32)
    if !inverse {
        value = value * signs
    }
    value = hadamardTransform(
        value.reshaped([-1, block]), scale: 1 / Float(block).squareRoot()
    ).reshaped(shape)
    if inverse {
        value = value * signs
    }
    return value.asType(dtype)
}

/// Packed placeholders for one projection of `outputDimensions` by `inputDimensions`.
///
/// Allocated rather than quantized from a dense tensor: the checkpoint supplies every
/// value, and quantizing an untrained weight would discard the scales it ships with.
private func hadamardPlaceholders(inputDimensions: Int, outputDimensions: Int, block: Int) throws
    -> (weight: MLXArray, scales: MLXArray, biases: MLXArray, signs: MLXArray?)
{
    guard inputDimensions % 128 == 0 else {
        throw PrismHadamardError.packedWidth(inputDimensions)
    }
    guard [0, 512, 1024, 2048, 4096].contains(block),
        block == 0 || inputDimensions % block == 0
    else {
        throw PrismHadamardError.blockSize(block: block, width: inputDimensions)
    }
    return (
        MLXArray.zeros([outputDimensions, inputDimensions / 16], dtype: .uint32),
        MLXArray.zeros([outputDimensions, inputDimensions / 128], dtype: .float32),
        MLXArray.zeros([outputDimensions, inputDimensions / 128], dtype: .float32),
        block == 0 ? nil : MLXArray.ones([inputDimensions], dtype: .float32)
    )
}

/// One packed projection: ordinary 2-bit affine weights over rotated activations.
///
/// Subclasses `Linear` for the same reason `QuantizedLinear` does: the model publishes its
/// projections through typed slots, and a replacement that is not a `Linear` cannot be set
/// into one.
private class HadamardQuantizedLinear: Linear, Quantized {
    let block: Int
    let bits = 2
    let groupSize = 128
    let mode = QuantizationMode.affine

    @ParameterInfo(key: "scales") var scales: MLXArray
    @ParameterInfo(key: "biases") var biases: MLXArray
    @ParameterInfo(key: "signs") var signs: MLXArray?

    init(inputDimensions: Int, outputDimensions: Int, block: Int) throws {
        let packed = try hadamardPlaceholders(
            inputDimensions: inputDimensions, outputDimensions: outputDimensions, block: block)
        self.block = block
        self._scales.wrappedValue = packed.scales
        self._biases.wrappedValue = packed.biases
        self._signs.wrappedValue = packed.signs
        super.init(weight: packed.weight, bias: nil)
        freeze()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        if let signs, block != 0 {
            x = hadamardRotate(x, block: block, signs: signs)
        }
        return quantizedMM(
            x, weight, scales: scales, biases: biases, transpose: true,
            groupSize: groupSize, bits: bits)
    }
}

/// The packed embedding: selected rows are dequantized and rotated back into its basis.
private class HadamardQuantizedEmbedding: Embedding, Quantized {
    let block: Int
    let bits = 2
    let groupSize = 128
    let mode = QuantizationMode.affine

    @ParameterInfo(key: "scales") var scales: MLXArray
    @ParameterInfo(key: "biases") var biases: MLXArray
    @ParameterInfo(key: "signs") var signs: MLXArray?

    init(inputDimensions: Int, outputDimensions: Int, block: Int) throws {
        let packed = try hadamardPlaceholders(
            inputDimensions: inputDimensions, outputDimensions: outputDimensions, block: block)
        self.block = block
        self._scales.wrappedValue = packed.scales
        self._biases.wrappedValue = packed.biases
        self._signs.wrappedValue = packed.signs
        super.init(weight: packed.weight)
        freeze()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let flat = x.reshaped([-1])
        var out = dequantized(
            weight[flat], scales: scales[flat], biases: biases[flat],
            groupSize: groupSize, bits: bits
        )
        .reshaped(shape + [-1])
        .asType(.float16)
        if let signs, block != 0 {
            out = hadamardRotate(out, block: block, signs: signs, inverse: true)
        }
        return out
    }

    override func asLinear(_ x: MLXArray) -> MLXArray {
        var x = x
        if let signs, block != 0 {
            x = hadamardRotate(x, block: block, signs: signs)
        }
        return quantizedMM(
            x, weight, scales: scales, biases: biases, transpose: true,
            groupSize: groupSize, bits: bits)
    }
}

/// A Hadamard pack that cannot be loaded as it stands.
public enum PrismHadamardError: Error, CustomStringConvertible {
    case schema(String)
    case packedWidth(Int)
    case blockSize(block: Int, width: Int)
    case unknownModule(String)
    case duplicateModule(String)
    case unsupportedModule(path: String, type: String)
    case kindMismatch(String)
    case activationDType(path: String, dtype: String)
    case outputBias(String)
    case invalidSigns(String)

    public var description: String {
        switch self {
        case .schema(let detail): detail
        case .packedWidth(let width):
            "packed input width \(width) is not divisible by 128"
        case .blockSize(let block, let width):
            "Hadamard block \(block) is invalid for packed input width \(width)"
        case .unknownModule(let path): "unknown packed module: \(path)"
        case .duplicateModule(let path): "duplicate packed module: \(path)"
        case .unsupportedModule(let path, let type):
            "unsupported packed module: \(path) is a \(type)"
        case .kindMismatch(let path): "packed module kind mismatch: \(path)"
        case .activationDType(let path, let dtype):
            "unsupported packed activation dtype \(dtype) for \(path)"
        case .outputBias(let path):
            "packed linear with output bias is unsupported: \(path)"
        case .invalidSigns(let path):
            "\(path) has Hadamard signs that are not all -1 or +1"
        }
    }
}

/// Ternary Bonsai 2: a Hadamard-rotated 2-bit pack of Qwen3.5, vision tower included.
///
/// The configuration is Qwen3.5's plus a manifest naming every projection the pack
/// replaced. Only schema 2 packs with grouped GDN activations are supported.
public struct PrismHadamardQwen35Configuration: Codable, Sendable,
    ModelConfigurationValidating
{

    /// One entry of the pack's manifest.
    public struct PackedModule: Codable, Sendable {
        public var path: String
        public var block: Int
        public var embedding: Bool
        public var dtype: String
    }

    public var qwen35: Qwen35Configuration
    public var modules: [PackedModule]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case baseModelType = "base_model_type"
        case tensorNamespace = "tensor_namespace"
        case gdnActivationLayout = "gdn_activation_layout"
        case quantization
        case modules
    }

    private struct Quantization: Codable, Sendable {
        var bits: Int
        var groupSize: Int
        var mode: String

        enum CodingKeys: String, CodingKey {
            case bits
            case groupSize = "group_size"
            case mode
        }
    }

    private var schemaVersion = 0
    private var baseModelType = ""
    private var tensorNamespace = ""
    private var gdnActivationLayout = ""
    private var quantization: Quantization?

    public init(from decoder: Decoder) throws {
        self.qwen35 = try Qwen35Configuration(from: decoder)
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.schemaVersion = try container.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 2
        self.baseModelType =
            try container.decodeIfPresent(String.self, forKey: .baseModelType) ?? "qwen3_5"
        self.tensorNamespace =
            try container.decodeIfPresent(String.self, forKey: .tensorNamespace)
            ?? "mlx-vlm-qwen3_5"
        self.gdnActivationLayout =
            try container.decodeIfPresent(String.self, forKey: .gdnActivationLayout) ?? "grouped"
        self.quantization = try container.decodeIfPresent(
            Quantization.self, forKey: .quantization)
        self.modules =
            try container.decodeIfPresent([PackedModule].self, forKey: .modules) ?? []
    }

    public func encode(to encoder: Encoder) throws {
        try qwen35.encode(to: encoder)
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(schemaVersion, forKey: .schemaVersion)
        try container.encode(baseModelType, forKey: .baseModelType)
        try container.encode(tensorNamespace, forKey: .tensorNamespace)
        try container.encode(gdnActivationLayout, forKey: .gdnActivationLayout)
        try container.encodeIfPresent(quantization, forKey: .quantization)
        try container.encode(modules, forKey: .modules)
    }

    public func validateModelConfiguration() throws {
        guard schemaVersion == 2, baseModelType == "qwen3_5" else {
            throw PrismHadamardError.schema(
                "only schema 2 Qwen3.5 Hadamard packs are supported, got schema "
                    + "\(schemaVersion) of \(baseModelType)")
        }
        guard tensorNamespace == "mlx-vlm-qwen3_5" else {
            throw PrismHadamardError.schema(
                "unsupported Hadamard tensor namespace: \(tensorNamespace)")
        }
        guard gdnActivationLayout == "grouped" else {
            throw PrismHadamardError.schema(
                "Hadamard packs require grouped GDN activations, got \(gdnActivationLayout)")
        }
        guard let quantization, quantization.bits == 2, quantization.groupSize == 128,
            quantization.mode == "affine"
        else {
            throw PrismHadamardError.schema(
                "Hadamard packs require 2-bit affine weights at group size 128")
        }
        guard !modules.isEmpty else {
            throw PrismHadamardError.schema(
                "Hadamard pack is missing its packed module manifest")
        }
    }
}

/// Qwen3.5 with the pack's projections swapped in.
public class PrismHadamardQwen35: Qwen35 {
    private let packedModules: [PrismHadamardQwen35Configuration.PackedModule]

    public init(_ configuration: PrismHadamardQwen35Configuration) throws {
        try configuration.validateModelConfiguration()
        self.packedModules = configuration.modules
        super.init(configuration.qwen35)

        let existing = Dictionary(
            languageModel.namedModules(), uniquingKeysWith: { first, _ in first })
        var replacements: [(String, Module)] = []
        var seen = Set<String>()

        for record in configuration.modules {
            guard seen.insert(record.path).inserted else {
                throw PrismHadamardError.duplicateModule(record.path)
            }
            guard let original = existing[record.path] else {
                throw PrismHadamardError.unknownModule(record.path)
            }
            guard record.dtype == "float16" else {
                throw PrismHadamardError.activationDType(path: record.path, dtype: record.dtype)
            }

            let rows: Int
            let width: Int
            if let linear = original as? Linear {
                guard !record.embedding else {
                    throw PrismHadamardError.kindMismatch(record.path)
                }
                guard linear.bias == nil else {
                    throw PrismHadamardError.outputBias(record.path)
                }
                (rows, width) = (linear.weight.dim(0), linear.weight.dim(1))
            } else if let embedding = original as? Embedding {
                guard record.embedding else {
                    throw PrismHadamardError.kindMismatch(record.path)
                }
                (rows, width) = (embedding.weight.dim(0), embedding.weight.dim(1))
            } else {
                throw PrismHadamardError.unsupportedModule(
                    path: record.path, type: "\(type(of: original))")
            }

            let replacement: Module =
                record.embedding
                ? try HadamardQuantizedEmbedding(
                    inputDimensions: width, outputDimensions: rows, block: record.block)
                : try HadamardQuantizedLinear(
                    inputDimensions: width, outputDimensions: rows, block: record.block)
            replacements.append((record.path, replacement))
        }

        try languageModel.update(
            modules: ModuleChildren.unflattened(replacements), verify: .noUnusedKeys)
    }

    /// Schema 2 tensors already have MLX layouts and offset RMSNorm weights.
    public override func sanitize(
        weights: [String: MLXArray], metadata: [String: String]
    ) -> [String: MLXArray] {
        sanitize(weights: weights)
    }

    public override func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        // Widen once at load to avoid dtype-conversion copies on every quantized matmul.
        for record in packedModules {
            for suffix in ["scales", "biases"] {
                let key = "language_model.\(record.path).\(suffix)"
                if let value = weights[key], value.dtype == .float16 {
                    weights[key] = value.asType(.float32)
                }
            }
        }
        return weights
    }

    /// Invalid signs must fail weight loading; inference preparation errors are recoverable.
    public override func update(
        parameters: ModuleParameters, verify: VerifyUpdate,
        path: [String] = [], modulePath: [String] = []
    ) throws -> Self {
        let weights = Dictionary(uniqueKeysWithValues: parameters.flattened())
        let checks = packedModules.compactMap { record -> (String, MLXArray)? in
            guard record.block != 0,
                let signs = weights["language_model.\(record.path).signs"]
            else { return nil }
            return (record.path, MLX.all((signs .== 1) .|| (signs .== -1)))
        }
        if !checks.isEmpty, !MLX.all(stacked(checks.map(\.1))).item(Bool.self) {
            let invalid = checks.first { !$0.1.item(Bool.self) }!
            throw PrismHadamardError.invalidSigns(invalid.0)
        }
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }
}
