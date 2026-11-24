import Foundation
import MLX
import MLXNN

/// Component configurations in Diffusers snapshots are independent JSON documents.
struct DiffusionConfiguration {
    let values: [String: Any]
    init(_ values: [String: Any] = [:]) { self.values = values }
    init(contentsOf url: URL) throws {
        guard
            let values = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [String: Any]
        else {
            throw ModelExecutionError.invalidOptions("Invalid component configuration: \(url.path)")
        }
        self.values = values
    }
    func int(_ key: String, _ fallback: Int) -> Int {
        (values[key] as? NSNumber)?.intValue ?? (values[key] as? String).flatMap(Int.init)
            ?? fallback
    }
    func float(_ key: String, _ fallback: Float) -> Float {
        (values[key] as? NSNumber)?.floatValue ?? (values[key] as? String).flatMap(Float.init)
            ?? fallback
    }
    func bool(_ key: String, _ fallback: Bool) -> Bool { values[key] as? Bool ?? fallback }
    func ints(_ key: String, _ fallback: [Int]) -> [Int] { values[key] as? [Int] ?? fallback }
    func string(_ key: String, _ fallback: String = "") -> String {
        values[key] as? String ?? fallback
    }
    func child(_ key: String) -> Self { .init(values[key] as? [String: Any] ?? [:]) }
}

struct FlowEulerSchedule {
    let sigmas: MLXArray
    init(steps: Int, shift: Float) {
        let base = linspace(1, 1 / Float(steps), count: steps)
        sigmas = concatenated([shift * base / (1 + (shift - 1) * base), MLXArray.zeros([1])])
    }
    func step(_ velocity: MLXArray, at index: Int, latents: MLXArray) -> MLXArray {
        latents + (sigmas[index + 1] - sigmas[index]).asType(latents.dtype)
            * velocity.asType(latents.dtype)
    }
}

func diffusionTimeEmbedding(_ t: MLXArray, dimensions: Int, sineFirst: Bool = false) -> MLXArray {
    let half = dimensions / 2
    let frequencies = exp(-log(Float(10000)) * arange(half).asType(.float32) / Float(half))
    let angles = t.reshaped([-1, 1]).asType(.float32) * frequencies
    return concatenated(
        sineFirst ? [sin(angles), cos(angles)] : [cos(angles), sin(angles)], axis: -1)
}

/// Angles for independent temporal, vertical and horizontal rotary axes.
func diffusionRotaryAngles(_ positions: MLXArray, sections: [Int], theta: Float) -> MLXArray {
    concatenated(
        sections.enumerated().map { axis, width in
            let frequencies = pow(
                MLXArray(theta), -arange(0, width, step: 2).asType(.float32) / Float(width))
            return positions[.ellipsis, axis].asType(.float32).expandedDimensions(axis: -1)
                * frequencies
        }, axis: -1)
}

func diffusionRotatePairs(_ x: MLXArray, angles: MLXArray) -> MLXArray {
    let pairs = x.reshaped(Array(x.shape.dropLast()) + [x.dim(-1) / 2, 2])
    let real = pairs[.ellipsis, 0]
    let imaginary = pairs[.ellipsis, 1]
    let c = cos(angles).expandedDimensions(axis: -2)
    let s = sin(angles).expandedDimensions(axis: -2)
    return stacked([real * c - imaginary * s, real * s + imaginary * c], axis: -1).reshaped(x.shape)
}

func diffusionPositions(
    frames: Int = 1, height: Int, width: Int, offset: Int = 0, centered: Bool = false
) -> MLXArray {
    var values = [Float]()
    for frame in 0 ..< frames {
        for row in 0 ..< height {
            for column in 0 ..< width {
                values += [
                    Float(frame + offset), Float(row - (centered ? height - height / 2 : 0)),
                    Float(column - (centered ? width - width / 2 : 0)),
                ]
            }
        }
    }
    return MLXArray(values, [1, frames * height * width, 3])
}

struct DiffusionCheckpoint {
    var weights: [String: MLXArray]
    let metadata: DiffusionConfiguration
    init(directory: URL) throws {
        let directory = directory.resolvingSymlinksInPath()
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "safetensors" && !$0.lastPathComponent.hasPrefix("._") }
        .sorted { $0.path < $1.path }
        guard !files.isEmpty else {
            throw ModelExecutionError.invalidOptions("Missing safetensors in \(directory.path)")
        }
        var weights = [String: MLXArray]()
        let index = directory.appending(path: "model.safetensors.index.json")
        var metadata =
            FileManager.default.fileExists(atPath: index.path)
            ? try DiffusionConfiguration(contentsOf: index).child("metadata").values : [:]
        for file in files {
            let (arrays, shardMetadata) = try loadArraysAndMetadata(url: file)
            for (key, value) in shardMetadata {
                metadata[key] =
                    (try? JSONSerialization.jsonObject(
                        with: Data(value.utf8), options: .fragmentsAllowed)) ?? value
            }
            for (key, value) in arrays {
                guard weights[key] == nil else {
                    throw ModelExecutionError.invalidOptions("Duplicate checkpoint tensor: \(key)")
                }
                weights[key] = value
            }
        }
        self.weights = weights
        if let config = try? DiffusionConfiguration(
            contentsOf: directory.appending(path: "config.json")),
            let quantization = config.values["quantization"] as? [String: Any]
        {
            metadata["quantization_mode"] =
                metadata["quantization_mode"] ?? quantization["mode"] ?? "affine"
            metadata["quantization_level"] = metadata["quantization_level"] ?? quantization["bits"]
            metadata["quantization_group_size"] =
                metadata["quantization_group_size"] ?? quantization["group_size"]
        }
        self.metadata = .init(metadata)
    }
    func apply(to model: Module, evaluate: Bool = true) throws {
        let paths = Set(
            weights.keys.filter { $0.hasSuffix(".scales") }.map { String($0.dropLast(7)) })
        if !paths.isEmpty {
            let candidates = [
                metadata, metadata.child("quantization"), metadata.child("quantization_config"),
            ]
            func value(_ names: [String]) -> String? {
                for c in candidates {
                    for key in names {
                        if let v = c.values[key] { return String(describing: v) }
                    }
                }
                return nil
            }
            let dimensions = Dictionary(
                uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
            let defaults: [String: (Int, Int)] = [
                "affine": (64, 4), "mxfp4": (32, 4), "nvfp4": (16, 4), "mxfp8": (32, 8),
            ]
            var modeName = value(["mode", "quantization_mode", "q_mode"])
            if modeName == nil {
                if paths.contains(where: { weights[$0 + ".biases"] != nil }) {
                    modeName = "affine"
                } else {
                    let compatible = defaults.filter { name, format in
                        name != "affine"
                            && paths.allSatisfy { path in
                                guard let shape = dimensions[path + ".weight"],
                                    let packed = weights[path + ".weight"],
                                    let scales = weights[path + ".scales"]
                                else { return false }
                                return packed.shape == Array(shape.dropLast()) + [
                                    (shape.last! * format.1 + 31) / 32
                                ]
                                    && scales.shape == Array(shape.dropLast()) + [
                                        shape.last! / format.0
                                    ]
                            }
                    }
                    if compatible.count == 1 { modeName = compatible.keys.first }
                }
            }
            guard let modeName, let mode = QuantizationMode(rawValue: modeName),
                let format = defaults[modeName]
            else {
                throw ModelExecutionError.invalidOptions(
                    "Quantized checkpoint requires an unambiguous supported quantization mode")
            }
            let bits =
                value(["bits", "num_bits", "quantization_level", "q_bits"]).flatMap(Int.init)
                ?? format.1
            let group =
                value(["group_size", "quantization_group_size", "q_group_size"]).flatMap(Int.init)
                ?? format.0
            guard [2, 3, 4, 5, 6, 8].contains(bits), [16, 32, 64, 128].contains(group),
                modeName == "affine" || (group == format.0 && bits == format.1)
            else {
                throw ModelExecutionError.invalidOptions(
                    "Unsupported quantization bits or group size")
            }
            for path in paths {
                guard let shape = dimensions[path + ".weight"],
                    let packed = weights[path + ".weight"], let scales = weights[path + ".scales"],
                    packed.dtype == .uint32, shape.last! % group == 0,
                    packed.shape == Array(shape.dropLast()) + [(shape.last! * bits + 31) / 32],
                    scales.shape == Array(shape.dropLast()) + [shape.last! / group],
                    (weights[path + ".biases"] != nil) == (modeName == "affine")
                else {
                    throw ModelExecutionError.invalidOptions(
                        "Invalid quantized weights or scales for \(path)")
                }
            }
            try quantizeByParent(model: model, groupSize: group, bits: bits, mode: mode) {
                path, _ in paths.contains(path)
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        if evaluate { eval(model) }
    }
}

extension Flux2TextEncoderConfiguration {
    init(diffusion c: DiffusionConfiguration) {
        self.init(
            vocabSize: c.int("vocab_size", 151936), hiddenSize: c.int("hidden_size", 2560),
            numHiddenLayers: c.int("num_hidden_layers", 36),
            numAttentionHeads: c.int("num_attention_heads", 32),
            numKeyValueHeads: c.int("num_key_value_heads", 8),
            intermediateSize: c.int("intermediate_size", 9728),
            maxPositionEmbeddings: c.int("max_position_embeddings", 40960),
            ropeTheta: c.float("rope_theta", 1e6),
            rmsNormEps: c.float("rms_norm_eps", 1e-6), headDim: c.int("head_dim", 128),
            attentionBias: c.bool("attention_bias", false))
    }
}

final class DiffusionRMSNorm: Module, UnaryLayer {
    let epsilon: Float
    init(_ epsilon: Float) {
        self.epsilon = epsilon
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.ones([x.dim(-1)], dtype: x.dtype), eps: epsilon)
    }
}

/// Separable resize filters for interleaved RGB byte images.
enum DiffusionRGBResize {
    static func rgbBytes(_ image: MLXArray, flattenAlpha: Bool = false) -> MLXArray {
        let pixels = (image.transposed(0, 2, 3, 1) + 1) / 2
        var rgb = pixels[.ellipsis, ..<3]
        if flattenAlpha, pixels.dim(-1) == 4 {
            let alpha = pixels[.ellipsis, 3...]
            rgb = rgb * alpha + (1 - alpha)
        }
        return clip(round(rgb * 255), min: 0, max: 255).asType(.uint8)
    }

    enum Filter {
        case lanczos, bicubic, pillowBicubic
    }

    static func resize(_ x: MLXArray, height: Int, width: Int, lanczos: Bool) -> MLXArray {
        resize(x, height: height, width: width, filter: lanczos ? .lanczos : .bicubic)
    }

    static func resize(_ x: MLXArray, height: Int, width: Int, filter: Filter) -> MLXArray {
        if x.dim(1) == height && x.dim(2) == width { return x }
        func quantize(_ values: MLXArray) -> MLXArray {
            clip(filter == .pillowBicubic ? floor(values + 0.5) : round(values), min: 0, max: 255)
        }
        var image = axis(x.asType(.float32), count: width, axis: 2, filter: filter)
        if filter != .bicubic { image = quantize(image) }
        return quantize(axis(image, count: height, axis: 1, filter: filter)).asType(.uint8)
    }

    private static func axis(_ image: MLXArray, count: Int, axis: Int, filter: Filter) -> MLXArray {
        let lanczos = filter == .lanczos
        let input = image.dim(axis)
        if input == count { return image }
        let scale = Float(input) / Float(count)
        let widening = max(1, scale)
        let radius = Int(ceil((lanczos ? 3 : 2) * widening))
        let centers = (arange(0, count, dtype: .float32) + 0.5) * scale - 0.5
        let offset = arange(1 - radius, radius + 1, dtype: .int32)
        let source = floor(centers).asType(.int32)[0..., .newAxis] + offset[.newAxis, 0...]
        let distance = (centers[0..., .newAxis] - source.asType(.float32)) / widening
        let absolute = abs(distance)
        let kernel: MLXArray
        if lanczos {
            let angle = Float.pi * distance
            let zero = absolute .< 1e-7
            let sinc = MLX.where(zero, 1, sin(angle) / angle)
            let window = MLX.where(zero, 1, sin(angle / 3) / (angle / 3))
            kernel = MLX.where(absolute .< 3, sinc * window, 0)
        } else {
            let a: Float = filter == .pillowBicubic ? -0.5 : -0.75
            let squared = square(absolute)
            let cubed = pow(absolute, 3)
            let first = (a + 2) * cubed - (a + 3) * squared + 1
            let second = a * cubed - 5 * a * squared + 8 * a * absolute - 4 * a
            kernel = MLX.where(absolute .< 1, first, MLX.where(absolute .< 2, second, 0))
        }
        let valid = (source .>= 0) .&& (source .< input)
        let weights = (kernel / widening) * valid
        let normalized = weights / weights.sum(axis: -1, keepDims: true)
        let gathered = take(image, clip(source, min: 0, max: input - 1), axis: axis)
        var shape = [1, 1, 1, 1, 1]
        shape[axis] = count
        shape[axis + 1] = 2 * radius
        return (gathered * normalized.reshaped(shape)).sum(axis: axis + 1)
    }
}
