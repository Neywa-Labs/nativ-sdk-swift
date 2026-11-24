import Foundation
import MLX
import MLXNN

enum ExpertScoring: Sendable {
    case sigmoid
    case softmax
}

/// Picks experts in two stages: groups first, then experts within the surviving groups.
///
/// The bias participates in *selection only* -- the scores returned come from the unbiased
/// values, so a correction term decides which experts run and not how much they count.
///
/// - Parameters:
///   - gates: raw router logits, `[..., experts]`.
///   - correctionBias: added before selection, broadcast over the expert axis.
///   - topK: experts kept per token.
///   - groups: expert groups; `1` skips the group stage entirely.
///   - topGroups: groups kept, scored by the sum of each group's two strongest experts.
///   - routedScalingFactor: multiplies the kept scores at the end.
///   - normalizeTopK: renormalize the kept scores to sum to one first.
/// - Returns: the selected indices and their scores.
func groupedExpertSelection(
    gates: MLXArray,
    correctionBias: MLXArray,
    topK: Int,
    groups: Int,
    topGroups: Int,
    routedScalingFactor: Float,
    normalizeTopK: Bool,
    scoring: ExpertScoring = .sigmoid,
    inFloat32: Bool = true
) -> (indices: MLXArray, scores: MLXArray) {
    let logits = inFloat32 ? gates.asType(.float32) : gates
    let originalScores =
        switch scoring {
        case .sigmoid: sigmoid(logits)
        case .softmax: softmax(logits, axis: -1, precise: !inFloat32)
        }

    var selection = originalScores + correctionBias.asType(originalScores.dtype)

    if groups > 1 {
        selection = unflatten(selection, axis: -1, shape: [groups, -1])
        let groupScores = top(selection, k: 2, axis: -1).sum(axis: -1, keepDims: true)

        let dropped = groups - topGroups
        if dropped > 0 {
            let groupIndices = argPartition(groupScores, kth: dropped - 1, axis: -2)[
                .ellipsis, ..<dropped, 0...]
            selection = putAlong(
                selection, stopGradient(groupIndices), values: MLXArray(-Float.infinity), axis: -2)
        }
        selection = flattened(selection, start: -2, end: -1)
    }

    let indices = argPartition(-selection, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
    var scores = takeAlong(originalScores, indices, axis: -1)

    if topK > 1, normalizeTopK {
        scores = scores / (scores.sum(axis: -1, keepDims: true) + 1e-20)
    }

    return (indices, scores * routedScalingFactor)
}
