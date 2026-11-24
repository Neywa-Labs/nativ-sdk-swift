import Foundation
import MLX

/// Inkling's banded relative-position mask, as one additive term.
///
/// Three things are folded into a single `[B, H, LQ, S]` array, which is why it is built by
/// hand rather than composed: causality, an optional sliding-window cap, and a learned
/// relative-position bias that applies only within `relExtent` positions of the query.
/// Beyond the band the bias is zero but attention is still allowed, so the band is not a
/// window -- the two are independent.
private let bandedMaskSource = """
        uint j  = thread_position_in_grid.x;   // key   position [0, S)
        uint i  = thread_position_in_grid.y;   // query position [0, LQ)
        uint bh = thread_position_in_grid.z;   // b * H + h
        if (i >= LQ || j >= S || bh >= B * H) return;
        uint b = bh / H, h = bh % H;
        int dist = (int(i) + int(Q_OFF)) - int(j);   // backward distance
        T val;
        if (dist < 0) {
            val = (T)(-1e30f);                                   // causal
        } else if (SLIDING > 0 && dist >= (int)SLIDING) {
            val = (T)(-1e30f);                                   // sliding-window cap
        } else if (dist < (int)REL_EXTENT) {
            float acc = 0.0f;
            uint rbase = ((b * LQ + i) * H + h) * D_REL;
            uint pcol = (uint)dist;
            for (uint d = 0; d < D_REL; ++d)
                acc += (float)rel[rbase + d] * (float)proj[d * REL_EXTENT + pcol];
            val = (T)acc;
        } else {
            val = (T)0;                                          // in-context, outside band
        }
        out[((b * H + h) * LQ + i) * S + j] = val;
    """

/// Shape-generic variant.
///
/// Every runtime dimension comes from injected shapes -- B, LQ and H from `rel`, and S from
/// a shape-carrying input that is never read -- and the query offset is `S - LQ`. Only the
/// per-layer constants stay template arguments, so one pipeline is compiled per layer kind
/// rather than one per `(LQ, S)` pair.
private let bandedMaskGenericSource = """
        const int B  = rel_shape[0];
        const int LQ = rel_shape[1];
        const int H  = rel_shape[2];
        const int S  = kshape_shape[2];
        uint j  = thread_position_in_grid.x;   // key   position [0, S)
        uint i  = thread_position_in_grid.y;   // query position [0, LQ)
        uint bh = thread_position_in_grid.z;   // b * H + h
        if ((int)i >= LQ || (int)j >= S || (int)bh >= B * H) return;
        uint b = bh / H, h = bh % H;
        int dist = ((int)i + (S - LQ)) - (int)j;     // backward distance
        T val;
        if (dist < 0) {
            val = (T)(-1e30f);                                   // causal
        } else if (SLIDING > 0 && dist >= (int)SLIDING) {
            val = (T)(-1e30f);                                   // sliding-window cap
        } else if (dist < (int)REL_EXTENT) {
            float acc = 0.0f;
            const size_t rbase = (size_t)b * rel_strides[0]
                + (size_t)i * rel_strides[1] + (size_t)h * rel_strides[2];
            for (uint d = 0; d < D_REL; ++d)
                acc += (float)rel[rbase + (size_t)d * rel_strides[3]]
                     * (float)proj[(size_t)d * proj_strides[0]
                                   + (size_t)dist * proj_strides[1]];
            val = (T)acc;
        } else {
            val = (T)0;                                          // in-context, outside band
        }
        out[(((size_t)b * H + h) * LQ + i) * S + j] = val;
    """

private final class BandedMaskKernels: @unchecked Sendable {
    static let shared = BandedMaskKernels()

    let templated: MLXFast.MLXFastKernel
    let generic: MLXFast.MLXFastKernel

    private init() {
        templated = MLXFast.metalKernel(
            name: "inkling_banded_mask",
            inputNames: ["rel", "proj"],
            outputNames: ["out"],
            source: bandedMaskSource)
        generic = MLXFast.metalKernel(
            name: "inkling_banded_mask_v2",
            inputNames: ["rel", "proj", "kshape"],
            outputNames: ["out"],
            source: bandedMaskGenericSource,
            ensureRowContiguous: false)
    }
}

private func roundUp(_ value: Int, _ multiple: Int) -> Int {
    ((value + multiple - 1) / multiple) * multiple
}

/// Builds the additive mask for one attention call.
///
/// - Parameters:
///   - rel: `[B, LQ, H, dRel]` per-query relative-position features.
///   - projection: `[dRel, relExtent]` learned bias projection.
///   - queryOffset: absolute position of the first query.
///   - keyLength: total key positions, `S`.
///   - slidingWindow: cap in positions, or zero for none.
///   - relativeExtent: how far the learned bias reaches.
///   - leftPadding: per-row count of padding positions to mask out.
///   - shapeReference: an array whose axis 2 is `S`, used only to carry that dimension into
///     the shape-generic kernel. Supplying it avoids recompiling per `(LQ, S)` pair.
/// - Returns: `[B, H, LQ, S]` additive mask in `rel`'s dtype.
func bandedAdditiveMask(
    rel: MLXArray,
    projection: MLXArray,
    queryOffset: Int,
    keyLength: Int,
    slidingWindow: Int,
    relativeExtent: Int,
    leftPadding: MLXArray? = nil,
    shapeReference: MLXArray? = nil
) -> MLXArray {
    let batch = rel.dim(0)
    let queryLength = rel.dim(1)
    let heads = rel.dim(2)
    let relDimensions = rel.dim(3)
    let dtype = rel.dtype
    let onGPU = Device.defaultDevice().deviceType == .gpu

    var mask: MLXArray
    if onGPU, let shapeReference, shapeReference.ndim >= 3,
        shapeReference.dim(2) == keyLength, queryOffset == keyLength - queryLength
    {
        mask =
            BandedMaskKernels.shared.generic(
                [rel, projection, shapeReference],
                template: [
                    ("T", dtype),
                    ("D_REL", relDimensions),
                    ("REL_EXTENT", relativeExtent),
                    ("SLIDING", slidingWindow),
                ],
                grid: (roundUp(keyLength, 8), roundUp(queryLength, 8), batch * heads),
                threadGroup: (8, 8, 1),
                outputShapes: [[batch, heads, queryLength, keyLength]],
                outputDTypes: [dtype]
            )[0]
    } else if onGPU {
        mask =
            BandedMaskKernels.shared.templated(
                [rel, projection],
                template: [
                    ("T", dtype),
                    ("B", batch),
                    ("H", heads),
                    ("LQ", queryLength),
                    ("S", keyLength),
                    ("Q_OFF", queryOffset),
                    ("D_REL", relDimensions),
                    ("REL_EXTENT", relativeExtent),
                    ("SLIDING", slidingWindow),
                ],
                grid: (roundUp(keyLength, 8), roundUp(queryLength, 8), batch * heads),
                threadGroup: (8, 8, 1),
                outputShapes: [[batch, heads, queryLength, keyLength]],
                outputDTypes: [dtype]
            )[0]
    } else {
        mask = bandedAdditiveMaskReference(
            rel: rel, projection: projection, queryOffset: queryOffset,
            keyLength: keyLength, slidingWindow: slidingWindow,
            relativeExtent: relativeExtent)
    }

    if let leftPadding {
        let valid =
            MLXArray(Int32(0) ..< Int32(keyLength))[.newAxis, 0...]
            .>= leftPadding[0..., .newAxis]
        mask = MLX.where(
            valid[0..., .newAxis, .newAxis, 0...], mask, MLXArray(-1e30, dtype: dtype))
    }
    return mask.asType(dtype)
}

/// The mask without a kernel, in plain array operations.
///
/// Upstream's CPU path, kept reachable on GPU too so the kernels can be checked against
/// something that does not depend on them.
func bandedAdditiveMaskReference(
    rel: MLXArray,
    projection: MLXArray,
    queryOffset: Int,
    keyLength: Int,
    slidingWindow: Int,
    relativeExtent: Int
) -> MLXArray {
    let batch = rel.dim(0)
    let queryLength = rel.dim(1)
    let heads = rel.dim(2)
    let dtype = rel.dtype

    let biases = rel.matmul(projection).transposed(0, 2, 1, 3)
    let queryPositions = MLXArray(Int32(0) ..< Int32(queryLength)) + Int32(queryOffset)
    let keyPositions = MLXArray(Int32(0) ..< Int32(keyLength))
    let distance = queryPositions[0..., .newAxis] - keyPositions[.newAxis, 0...]

    // Inside the band the bias is looked up by distance; the clip only keeps the gather in
    // range, and the positions it folds onto are overwritten below.
    let gathered = broadcast(
        clip(distance, min: 0, max: relativeExtent - 1)[.newAxis, .newAxis],
        to: [batch, heads, queryLength, keyLength])
    var bias = takeAlong(biases, gathered, axis: -1)
    bias = MLX.where(
        (distance .>= Int32(relativeExtent))[.newAxis, .newAxis],
        MLXArray(0, dtype: dtype), bias)

    var masked = distance .< 0
    if slidingWindow > 0 {
        masked = masked .|| (distance .>= Int32(slidingWindow))
    }
    return MLX.where(masked[.newAxis, .newAxis], MLXArray(-1e30, dtype: dtype), bias)
        .asType(dtype)
}
