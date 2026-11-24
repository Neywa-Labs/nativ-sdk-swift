import Foundation
import MLX
import MLXNN

/// Short-block affine GEMV with exactly the reductions used by singleton decode.
/// Unsupported layouts/formats retain the ordinary per-token fallback.
private func supportsExactQuantizedProjection(_ layer: QuantizedLinear, _ x: MLXArray) -> Bool {
    guard type(of: layer) == QuantizedLinear.self, Device.defaultDevice().deviceType == .gpu,
        x.ndim == 3, x.dim(0) > 0, (1 ... 16).contains(x.dim(1)),
        layer.mode == .affine, [4, 5, 8].contains(layer.bits),
        [32, 64, 128].contains(layer.groupSize),
        layer.weight.ndim == 2, layer.weight.dtype == .uint32,
        let biases = layer.biases, layer.scales.ndim == 2, biases.shape == layer.scales.shape,
        x.dtype == .float16 || x.dtype == .bfloat16,
        layer.scales.dtype == x.dtype, biases.dtype == x.dtype
    else { return false }
    let k = layer.weight.dim(1) * 32 / layer.bits
    let n = layer.weight.dim(0)
    return k > 0 && k % 512 == 0 && n > 0 && n % 8 == 0 && x.dim(2) == k
        && layer.scales.shape == [n, k / layer.groupSize]
}

private final class ExactQuantizedKernels: @unchecked Sendable {
    static let shared = ExactQuantizedKernels()
    enum Style: String { case standard, streamed, tiled }
    struct Key: Hashable {
        let bits: Int
        let group: Int
        let style: Style
        let fusedWidths: [Int]
    }
    private let lock = NSLock()
    private var kernels: [Key: MLXFast.MLXFastKernel] = [:]

    func kernel(_ key: Key) -> MLXFast.MLXFastKernel {
        lock.withLock {
            if let kernel = kernels[key] { return kernel }
            var source: String
            switch key.style {
            case .standard: source = ExactQuantizedKernelSource.standard
            case .streamed: source = ExactQuantizedKernelSource.streamed
            case .tiled: source = ExactQuantizedKernelSource.tiled
            }
            var names = ["x", "w", "scales", "biases"]
            if !key.fusedWidths.isEmpty {
                names =
                    ["x"]
                    + key.fusedWidths.indices.flatMap { ["w\($0)", "scales\($0)", "biases\($0)"] }
                var selection = """
                    int global_out = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
                    int local_out = global_out;
                    const device uint8_t* selected_w = (const device uint8_t*)w0;
                    const device T* selected_scales = scales0;
                    const device T* selected_biases = biases0;
                    """
                var offset = key.fusedWidths[0]
                for index in 1 ..< key.fusedWidths.count {
                    selection += """

                        if (global_out >= \(offset)) {
                            local_out = global_out - \(offset);
                            selected_w = (const device uint8_t*)w\(index);
                            selected_scales = scales\(index);
                            selected_biases = biases\(index);
                        }
                        """
                    offset += key.fusedWidths[index]
                }
                source = source.replacingOccurrences(
                    of: "int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;",
                    with: selection
                ).replacingOccurrences(
                    of: "(const device uint8_t*)w + out_row * in_vec_size_w",
                    with: "selected_w + local_out * in_vec_size_w"
                ).replacingOccurrences(
                    of: "scales + out_row * in_vec_size_g",
                    with: "selected_scales + local_out * in_vec_size_g"
                ).replacingOccurrences(
                    of: "biases + out_row * in_vec_size_g",
                    with: "selected_biases + local_out * in_vec_size_g"
                ).replacingOccurrences(
                    of: "int n = out_row + row;", with: "int n = global_out + row;")
            }
            let header = ExactQuantizedKernelSource.header
                .replacingOccurrences(of: "__BITS__", with: String(key.bits))
                .replacingOccurrences(of: "__GS__", with: String(key.group))
                .replacingOccurrences(
                    of: "__RESULTS_PER_SIMDGROUP__", with: key.style == .streamed ? "1" : "4")
            let suffix = key.fusedWidths.map(String.init).joined(separator: "_")
            let kernel = MLXFast.metalKernel(
                name:
                    "nativ_exact_affine_\(key.style.rawValue)_b\(key.bits)_g\(key.group)_\(suffix)",
                inputNames: names, outputNames: ["y"], source: source, header: header)
            kernels[key] = kernel
            return kernel
        }
    }
}

func exactQuantizedProjection(_ layer: Linear, _ x: MLXArray) -> MLXArray? {
    guard let layer = layer as? QuantizedLinear, supportsExactQuantizedProjection(layer, x) else {
        return nil
    }
    let length = x.dim(1)
    let style: ExactQuantizedKernels.Style =
        layer.bits == 4 && length >= 6
        ? (length <= 8 ? .streamed : .tiled) : .standard
    let kernel = ExactQuantizedKernels.shared.kernel(
        .init(
            bits: layer.bits, group: layer.groupSize, style: style, fusedWidths: []))
    let rows = style == .streamed ? 2 : 8
    let n = layer.weight.dim(0)
    let output = kernel(
        [contiguous(x), layer.weight, layer.scales, layer.biases!],
        template: [("T", x.dtype), ("VERIFY_T", length), ("K_SIZE", x.dim(2)), ("N_SIZE", n)],
        grid: (32, 2 * (n / rows), x.dim(0) * (style == .tiled ? (length + 1) / 2 : 1)),
        threadGroup: (32, 2, 1), outputShapes: [[x.dim(0), length, n]], outputDTypes: [x.dtype])[0]
    return layer.bias.map { output + $0 } ?? output
}

/// Fuse independent projections sharing the same activation block and quantization.
func exactQuantizedProjections(_ layers: [Linear], _ x: MLXArray) -> [MLXArray]? {
    guard (2 ... 4).contains(layers.count), x.ndim == 3, (2 ... 8).contains(x.dim(1)) else {
        return nil
    }
    let quantized = layers.compactMap { $0 as? QuantizedLinear }
    guard quantized.count == layers.count, let first = quantized.first,
        quantized.allSatisfy({
            $0.bias == nil && $0.bits == first.bits && $0.groupSize == first.groupSize
                && supportsExactQuantizedProjection($0, x)
        })
    else { return nil }
    let widths = quantized.map { $0.weight.dim(0) }
    let n = widths.reduce(0, +)
    let style: ExactQuantizedKernels.Style =
        first.bits == 4 && x.dim(1) >= 6 ? .streamed : .standard
    let kernel = ExactQuantizedKernels.shared.kernel(
        .init(
            bits: first.bits, group: first.groupSize, style: style, fusedWidths: widths))
    let result = kernel(
        [contiguous(x)] + quantized.flatMap { [$0.weight, $0.scales, $0.biases!] },
        template: [("T", x.dtype), ("VERIFY_T", x.dim(1)), ("K_SIZE", x.dim(2)), ("N_SIZE", n)],
        grid: (32, 2 * (n / (style == .streamed ? 2 : 8)), x.dim(0)),
        threadGroup: (32, 2, 1), outputShapes: [[x.dim(0), x.dim(1), n]], outputDTypes: [x.dtype])[
            0]
    var offset = 0
    let splits = widths.dropLast().map { width in
        offset += width
        return offset
    }
    return result.split(indices: splits, axis: -1)
}
