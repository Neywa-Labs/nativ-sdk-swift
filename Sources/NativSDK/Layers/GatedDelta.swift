//
//  GatedDelta.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py
//

import Foundation
import MLX
import MLXNN

// MARK: - Compute G

/// Fused form of the decay gate chain — elementwise, and MLX `compile`
/// preserves per-node dtype rounding (verified bitwise against the unfused
/// chain on the real decode/prefill shapes, bf16 and f16), so this is
/// bit-identical while cutting ~6 kernel launches per GDN layer per step.
let computeGatedDeltaG: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { aLog, a, dtBias in
    exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
}

// MARK: - Metal Kernel

private func makeGatedDeltaKernel(hasMask: Bool, history: Bool = false, vectorGate: Bool = false)
    -> MLXFast.MLXFastKernel?
{
    let maskSource = hasMask ? "mask[b_idx * T + t]" : "true"
    let gateOffset = vectorGate ? "(b_idx * T * Hv + hv_idx) * Dk" : "b_idx * T * Hv"
    let gateIndex = vectorGate ? "s_idx" : "hv_idx"
    let gateStride = vectorGate ? "Hv * Dk" : "Hv"

    let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            // q, k: [B, T, Hk, Dk]
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // g: [B, T, Hv] or [B, T, Hv, Dk]
            auto g_ = g + \(gateOffset);
            auto beta_ = beta + b_idx * T * Hv;

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            \(history ? "int history_index = 0;" : "")
            for (int t = 0; t < T; ++t) {
              if (\(maskSource)) {
                float kv_mem = 0.0f;
                {
                  // Preserve Kahan summation under Metal's default fast math.
                  #pragma clang fp reassociate(off)
                  #pragma clang fp contract(off)
                  float kv_compensation = 0.0f;
                  for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] * g_[\(gateIndex)];
                    auto product = state[i] * k_[s_idx];
                    auto corrected = product - kv_compensation;
                    auto next_sum = kv_mem + corrected;
                    kv_compensation = (next_sum - kv_mem) - corrected;
                    kv_mem = next_sum;
                  }
                }
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] + k_[s_idx] * delta;
                  out += state[i] * q_[s_idx];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {
                  y[dv_idx] = static_cast<InT>(out);
                }
              } else {
                y[dv_idx] = static_cast<InT>(0);
              }
              \(history ? """
              if (history_index < N && t + 1 == positions[history_index]) {
                auto saved = states + ((b_idx * N + history_index) * Hv + hv_idx) * Dv * Dk + dv_idx * Dk;
                for (int i = 0; i < n_per_t; ++i) {
                  saved[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
                }
                ++history_index;
              }
              """ : "")
              // Increment data pointers to next time step
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += \(gateStride);
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<StT>(state[i]);
            }
        """

    var inputNames = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if hasMask {
        inputNames.append("mask")
    }

    if history { inputNames += ["positions", "N"] }
    let suffix =
        (hasMask ? "_mask" : "") + (history ? "_history" : "") + (vectorGate ? "_vector" : "")

    return MLXFast.metalKernel(
        name: "gated_delta_step\(suffix)",
        inputNames: inputNames,
        outputNames: history ? ["y", "state_out", "states"] : ["y", "state_out"],
        source: source
    )
}

private final class GatedDeltaKernelManager: Sendable {
    static let shared = GatedDeltaKernelManager()

    let kernel: MLXFast.MLXFastKernel?
    let kernelMasked: MLXFast.MLXFastKernel?
    let vectorKernel: MLXFast.MLXFastKernel?
    let vectorKernelMasked: MLXFast.MLXFastKernel?
    let historyKernel: MLXFast.MLXFastKernel?
    let historyKernelMasked: MLXFast.MLXFastKernel?

    private init() {
        kernel = makeGatedDeltaKernel(hasMask: false)
        kernelMasked = makeGatedDeltaKernel(hasMask: true)
        vectorKernel = makeGatedDeltaKernel(hasMask: false, vectorGate: true)
        vectorKernelMasked = makeGatedDeltaKernel(hasMask: true, vectorGate: true)
        historyKernel = makeGatedDeltaKernel(hasMask: false, history: true)
        historyKernelMasked = makeGatedDeltaKernel(hasMask: true, history: true)
    }
}

// MARK: - Kernel Dispatch

func gatedDeltaKernel(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = k.dim(0)
    let T = k.dim(1)
    let Hk = k.dim(2)
    let Dk = k.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    let inputType = q.dtype
    let stateType = state.dtype

    let kernels = GatedDeltaKernelManager.shared
    let vectorGate = g.ndim == 4
    let selectedKernel =
        if vectorGate { mask == nil ? kernels.vectorKernel : kernels.vectorKernelMasked } else {
            mask == nil ? kernels.kernel : kernels.kernelMasked
        }
    var inputs: [MLXArray] = [q, k, v, g, beta, state, MLXArray(T)]
    if let mask {
        inputs.append(mask)
    }

    guard let kernel = selectedKernel else {
        fatalError("Gated delta kernel not available")
    }

    let outputs = kernel(
        inputs,
        template: [
            ("InT", inputType),
            ("StT", stateType),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid: (32, Dv, B * Hv),
        threadGroup: (32, 4, 1),
        outputShapes: [[B, T, Hv, Dv], state.shape],
        outputDTypes: [inputType, stateType]
    )

    return (outputs[0], outputs[1])
}

// MARK: - Ops Fallback

private func gatedDeltaStepOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let oldState = state
    let decay: MLXArray
    if g.ndim == 2 {
        decay = expandedDimensions(g, axes: [2, 3])
    } else if g.ndim == 3 {
        decay = expandedDimensions(g, axis: -2)
    } else {
        fatalError("Unsupported gating shape \(g.shape)")
    }

    var state = state * decay
    let kvMem = (state * expandedDimensions(k, axis: -2)).sum(axis: -1)
    let delta = (v - kvMem) * expandedDimensions(beta, axis: -1)
    state = state + expandedDimensions(k, axis: -2) * expandedDimensions(delta, axis: -1)
    let y = (state * expandedDimensions(q, axis: -2)).sum(axis: -1)

    if let mask {
        let expandedMask: MLXArray
        if mask.ndim == 1 {
            expandedMask = expandedDimensions(mask, axes: [1, 2, 3])
        } else if mask.ndim == 2 {
            expandedMask = expandedDimensions(mask, axes: [2, 3])
        } else if mask.ndim == 3 {
            expandedMask = expandedDimensions(mask, axis: -1)
        } else {
            fatalError("Unsupported mask shape \(mask.shape)")
        }
        state = MLX.where(expandedMask, state, oldState)
    }

    return (y.asType(q.dtype), state)
}

/// Steps per recompute chunk in `gatedDeltaOps`.
///
/// The ops path is the one that can be trained through (the fused kernel
/// has no gradient), and a recurrence run as plain ops keeps every step's
/// state for the backward pass: at Qwen3.5-27B's shapes (48 heads of
/// 128x128 fp32) that is about 4 MB a step, gigabytes per layer at a
/// thousand tokens. Each chunk is run through a custom function whose
/// backward recomputes the chunk instead, so only the chunk boundaries are
/// kept - what `mx.checkpoint` does for the Python trainer. Inference is
/// unaffected: the forward is the same ops in the same order.
let gatedDeltaRecomputeChunk = 16

private enum GatedDeltaRecompute {
    /// `inputs` = [q, k, v, g, beta, state] plus the mask when there is one,
    /// each sliced to the chunk; returns [y, state].
    static func steps(_ inputs: [MLXArray], masked: Bool) -> [MLXArray] {
        let (q, k, v, g, beta) = (inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])
        var state = inputs[5]
        let mask = masked ? inputs[6] : nil
        var ys = [MLXArray]()
        ys.reserveCapacity(q.dim(1))
        for t in 0 ..< q.dim(1) {
            let (y, newState) = gatedDeltaStepOps(
                q: q[0..., t],
                k: k[0..., t],
                v: v[0..., t],
                g: g[0..., t],
                beta: beta[0..., t],
                state: state,
                mask: mask.map { $0[0..., t] }
            )
            ys.append(y)
            state = newState
        }
        return [MLX.stacked(ys, axis: 1), state]
    }

    static let unmasked: @Sendable ([MLXArray]) -> [MLXArray] = CustomFunction {
        Forward { steps($0, masked: false) }
        VJP { primals, cotangents in
            vjp({ steps($0, masked: false) }, primals: primals, cotangents: cotangents).1
        }
    }

    static let masked: @Sendable ([MLXArray]) -> [MLXArray] = CustomFunction {
        Forward { steps($0, masked: true) }
        VJP { primals, cotangents in
            vjp({ steps($0, masked: true) }, primals: primals, cotangents: cotangents).1
        }
    }
}

func gatedDeltaOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = q.dim(0)
    let T = q.dim(1)
    let Hk = q.dim(2)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    var q = q
    var k = k

    let repeatFactor = Hv / Hk
    if repeatFactor > 1 {
        q = repeated(q, count: repeatFactor, axis: -2)
        k = repeated(k, count: repeatFactor, axis: -2)
    }

    var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)

    var ys = [MLXArray]()
    ys.reserveCapacity((T + gatedDeltaRecomputeChunk - 1) / gatedDeltaRecomputeChunk)

    for start in stride(from: 0, to: T, by: gatedDeltaRecomputeChunk) {
        let steps = start ..< min(start + gatedDeltaRecomputeChunk, T)
        var inputs = [
            q[0..., steps], k[0..., steps], v[0..., steps], g[0..., steps], beta[0..., steps],
            state,
        ]
        let run: ([MLXArray]) -> [MLXArray]
        if let mask {
            inputs.append(mask[0..., steps])
            run = GatedDeltaRecompute.masked
        } else {
            run = GatedDeltaRecompute.unmasked
        }
        let out = run(inputs)
        ys.append(out[0])
        state = out[1]
    }

    let y = ys.count == 1 ? ys[0] : MLX.concatenated(ys, axis: 1)
    return (y, state)
}

// MARK: - Public API

public func gatedDeltaUpdate(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil,
    useKernel: Bool = true,
    lowerBound: Float? = nil
) -> (MLXArray, MLXArray) {
    let beta = sigmoid(b).asType(.float32)
    let g =
        lowerBound.map {
            exp($0 * sigmoid(exp(aLog.asType(.float32)) * (a + dtBias)))
        } ?? computeGatedDeltaG(aLog, a, dtBias)

    let B = q.dim(0)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    // State kept in fp32 to match Python mlx-lm. Using q.dtype (bf16) loses
    // precision across T-step recurrence, compounding rounding error.
    var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)
    if state.dtype != .float32 {
        state = state.asType(.float32)
    }

    // The fused kernel distributes Dk over exactly 32 simd lanes
    // (`n_per_t = Dk / 32`, integer-truncating). A key head dim that is not a
    // multiple of 32 would silently drop its trailing `Dk % 32` state
    // dimensions and produce wrong output with no error. Stock Qwen3.5 uses
    // Dk = 192 (a multiple of 32), but the value is config-driven, so route any
    // non-multiple-of-32 Dk to the ops fallback, which handles an arbitrary key
    // dimension correctly (slower, but never truncating).
    //
    // `useKernel: false` is for training: the kernel is a custom Metal
    // kernel and has no gradient, so a model in training mode passes
    // `!training` here, as the Python model does with `use_kernel`.
    if useKernel, g.ndim == 3, GatedDeltaKernelManager.shared.kernel != nil, Dk % 32 == 0 {
        return gatedDeltaKernel(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
    }

    return gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
}

/// Same recurrence and reductions as ordinary decode, with selected post-token
/// states written by the kernel. No layer replay is needed after rejection.
func gatedDeltaUpdateWithHistory(
    q: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray, b: MLXArray,
    aLog: MLXArray, dtBias: MLXArray, state: MLXArray?, mask: MLXArray?,
    useKernel: Bool, positions: [Int]
) -> (output: MLXArray, state: MLXArray, history: MLXArray?) {
    guard !positions.isEmpty else {
        let (output, final) = gatedDeltaUpdate(
            q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias,
            state: state, mask: mask, useKernel: useKernel)
        return (output, final, nil)
    }
    let batch = q.dim(0)
    let length = q.dim(1)
    let hk = q.dim(2)
    let dk = q.dim(3)
    let hv = v.dim(2)
    let dv = v.dim(3)
    precondition(
        positions == positions.sorted() && Set(positions).count == positions.count
            && positions.allSatisfy { $0 > 0 && $0 < length })
    let initial = (state ?? MLXArray.zeros([batch, hv, dv, dk], dtype: .float32)).asType(.float32)
    let g = computeGatedDeltaG(aLog, a, dtBias)
    let beta = sigmoid(b).asType(.float32)
    let manager = GatedDeltaKernelManager.shared
    if useKernel, dk % 32 == 0,
        let kernel = mask == nil ? manager.historyKernel : manager.historyKernelMasked
    {
        var inputs = [q, k, v, g, beta, initial, MLXArray(length)]
        if let mask { inputs.append(mask) }
        inputs += [MLXArray(positions.map(Int32.init)), MLXArray(positions.count)]
        let outputs = kernel(
            inputs,
            template: [
                ("InT", q.dtype), ("StT", initial.dtype), ("Dk", dk), ("Dv", dv), ("Hk", hk),
                ("Hv", hv),
            ],
            grid: (32, dv, batch * hv), threadGroup: (32, 4, 1),
            outputShapes: [
                [batch, length, hv, dv], initial.shape, [batch, positions.count, hv, dv, dk],
            ],
            outputDTypes: [q.dtype, initial.dtype, initial.dtype])
        return (outputs[0], outputs[1], outputs[2])
    }
    // Unusual head dimensions and training retain the ordinary ops path.
    var current = initial
    var output: [MLXArray] = []
    var history: [MLXArray] = []
    var start = 0
    for end in positions + [length] {
        let (part, next) = gatedDeltaUpdate(
            q: q[0..., start ..< end], k: k[0..., start ..< end], v: v[0..., start ..< end],
            a: a[0..., start ..< end], b: b[0..., start ..< end],
            aLog: aLog, dtBias: dtBias, state: current,
            mask: mask.map { $0[0..., start ..< end] }, useKernel: useKernel)
        output.append(part)
        current = next
        if end < length { history.append(current) }
        start = end
    }
    return (concatenated(output, axis: 1), current, stacked(history, axis: 1))
}
