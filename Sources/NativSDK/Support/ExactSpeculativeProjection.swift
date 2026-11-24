// Exact short-block GEMV accumulation mirrors mlx-vlm's verifier and MLX's singleton kernel.
import MLX
import MLXNN

private let exactVerifyKernel = MLXFast.metalKernel(
    name: "nativ_exact_speculative_verify_gemv",
    inputNames: ["x", "weight"], outputNames: ["out"],
    source: """
        uint lane = thread_position_in_grid.x;
                uint out_block = thread_position_in_grid.y;
                uint row = thread_position_in_grid.z;

                constexpr int TM = 4;
                constexpr int TN = 4;
                constexpr int SN = 32;
                constexpr int blockN = SN * TN;

                if (row >= R) {
                    return;
                }

                int out_row = int(out_block * TM);
                if (out_row >= O) {
                    return;
                }

                const device T* in_vec = x + row * K;
                const device T* mat = weight + out_row * K;

                float result[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
                int col = int(lane * TN);
                int n_iter = K / blockN;
                int leftover = K - blockN * n_iter;

                for (int iter = 0; iter < n_iter; ++iter) {
                    float v[TN];
                    for (int tn = 0; tn < TN; ++tn) {
                        v[tn] = static_cast<float>(in_vec[col + tn]);
                    }

                    for (int tm = 0; tm < TM; ++tm) {
                        for (int tn = 0; tn < TN; ++tn) {
                            result[tm] +=
                                static_cast<float>(mat[tm * K + col + tn]) * v[tn];
                        }
                    }

                    col += blockN;
                }

                if (leftover > 0) {
                    float v[TN];
                    for (int tn = 0; tn < TN; ++tn) {
                        v[tn] =
                            (col + tn < K) ? static_cast<float>(in_vec[col + tn]) : 0.0f;
                    }

                    for (int tm = 0; tm < TM; ++tm) {
                        for (int tn = 0; tn < TN; ++tn) {
                            T m =
                                (col + tn < K) ? mat[tm * K + col + tn] : T(0);
                            result[tm] += static_cast<float>(m) * v[tn];
                        }
                    }
                }

                for (int tm = 0; tm < TM; ++tm) {
                    for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                        result[tm] += simd_shuffle_down(result[tm], sn);
                    }
                }

                if (lane == 0) {
                    for (int tm = 0; tm < TM; ++tm) {
                        out[row * O + out_row + tm] = static_cast<T>(result[tm]);
                    }
                }
        """,
    header: "#include <metal_simdgroup>\nusing namespace metal;\n"
)

func exactSpeculativeProjection(weight: MLXArray, input x: MLXArray) -> MLXArray? {
    guard Device.defaultDevice().deviceType == .gpu,
        x.ndim == 3, x.dim(1) > 1, weight.ndim == 2,
        weight.dtype == x.dtype, x.dtype == .bfloat16 || x.dtype == .float16,
        weight.dim(0) >= 4, weight.dim(0) % 4 == 0,
        x.dim(-1) == weight.dim(1), x.dim(-1) < 16 * weight.dim(0)
    else { return nil }
    let rows = x.dim(0) * x.dim(1)
    let outputs = weight.dim(0)
    let out = exactVerifyKernel(
        [x.reshaped(rows, x.dim(-1)), weight],
        template: [("T", x.dtype), ("K", x.dim(-1)), ("O", outputs), ("R", rows)],
        grid: (32, outputs / 4, ((rows + 7) / 8) * 8), threadGroup: (32, 1, 8),
        outputShapes: [[rows, outputs]], outputDTypes: [x.dtype]
    )[0]
    return out.reshaped(x.dim(0), x.dim(1), outputs)
}

func exactSpeculativeLinear(_ linear: Linear, _ x: MLXArray) -> MLXArray {
    if x.dim(1) > 1, let out = exactQuantizedProjection(linear, x) { return out }
    if type(of: linear) == Linear.self,
        let out = exactSpeculativeProjection(weight: linear.weight, input: x)
    {
        return linear.bias.map { out + $0 } ?? out
    }
    if x.dim(1) > 1 {
        return concatenated(
            (0 ..< x.dim(1)).map { linear(x[0..., $0 ..< ($0 + 1), 0...]) }, axis: 1)
    }
    return linear(x)
}

func exactSpeculativeLinears(_ linears: [Linear], _ x: MLXArray) -> [MLXArray] {
    exactQuantizedProjections(linears, x) ?? linears.map { exactSpeculativeLinear($0, x) }
}
