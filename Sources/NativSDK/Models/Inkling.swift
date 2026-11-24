import Foundation
import MLX
import MLXNN

/// Advances a short convolution's state by the tokens just seen.
///
/// With no mask this is the trailing `stateSize` rows of `state` followed by `inputs`.
/// With one, only the valid span of each row counts, so the window is gathered per row:
/// the state's tail when the row has fewer valid tokens than the window, and the inputs
/// from the first valid position otherwise. Left-padded rows would otherwise fold padding
/// into their convolution state and carry it forward for the rest of the sequence.
func nextInklingConvolutionState(
    state: MLXArray, inputs: MLXArray, mask: MLXArray?
) -> MLXArray {
    let stateSize = state.dim(1)
    guard stateSize != 0 else { return state }
    guard let mask else {
        return concatenated([state, inputs], axis: 1)[0..., (-stateSize)..., 0...]
    }

    let validCount = mask.sum(axis: 1).asType(.int32)
    let validStart = argMax(mask.asType(.int32), axis: 1)
    // Row-relative position of each slot: negative reaches back into the state.
    let logical =
        MLXArray(Int32(0) ..< Int32(stateSize))[.newAxis, 0...] + validCount[0..., .newAxis]
        - Int32(stateSize)
    let stateIndices = Int32(stateSize) + logical
    let inputIndices =
        Int32(stateSize) + validStart[0..., .newAxis] + maximum(logical, 0)
    var indices = MLX.where(logical .>= 0, inputIndices, stateIndices)
    let combined = concatenated([state, inputs], axis: 1)
    indices = broadcast(
        indices[.ellipsis, .newAxis], to: indices.shape + [inputs.dim(-1)])
    return takeAlong(combined, indices, axis: 1)
}

/// Fused decode path for a four-tap convolution.
///
/// The rounding is deliberate and load-bearing: the convolution emits the layer dtype
/// before the fp32 residual add, and the layer's own residual is a second add on top of
/// that. Doing both in fp32 and rounding once would not match the unfused path.
private let inklingShortConvolutionSource = """
        uint c = thread_position_in_grid.x;   // channel
        uint b = thread_position_in_grid.y;   // batch row
        if (c >= C || b >= B) return;
        float w0 = (float)w[c * K + 0];
        float w1 = (float)w[c * K + 1];
        float w2 = (float)w[c * K + 2];
        float w3 = (float)w[c * K + 3];
        // Virtual padded input xp = [state (K-1 rows, fp32); x (L rows, T)].
        for (uint i = 0; i < L; ++i) {
            float acc = 0.0f;
            for (uint k = 0; k < K; ++k) {
                int r = (int)(i + k) - (int)(K - 1);  // row into x; negative -> state
                float v = (r < 0)
                    ? state[(b * (K - 1) + (uint)(r + (int)(K - 1))) * C + c]
                    : (float)x[(b * L + (uint)r) * C + c];
                float wk = (k == 0) ? w0 : (k == 1) ? w1 : (k == 2) ? w2 : w3;
                acc += wk * v;
            }
            float conv_r = (float)((T)acc);
            T inner = (T)(conv_r + (float)x[(b * L + i) * C + c]);
            if (HAS_RES) {
                out[(b * L + i) * C + c] =
                    (T)((float)inner + (float)res[(b * L + i) * C + c]);
            } else {
                out[(b * L + i) * C + c] = inner;
            }
        }
        for (uint s = 0; s < K - 1; ++s) {
            int r = (int)(L + s) - (int)(K - 1);
            nstate[(b * (K - 1) + s) * C + c] = (r < 0)
                ? state[(b * (K - 1) + (uint)(r + (int)(K - 1))) * C + c]
                : (float)x[(b * L + (uint)r) * C + c];
        }
    """

private final class InklingShortConvolutionKernel: @unchecked Sendable {
    static let shared = InklingShortConvolutionKernel()
    let kernel: MLXFast.MLXFastKernel

    private init() {
        kernel = MLXFast.metalKernel(
            name: "inkling_sconv_decode",
            inputNames: ["x", "state", "w", "res"],
            outputNames: ["out", "nstate"],
            source: inklingShortConvolutionSource)
    }
}

/// Depthwise causal 1-D convolution over the previous `kernelSize - 1` positions, plus a
/// residual add.
///
/// Held in fp32 for stability, matching the reference. `convolutionIndex` selects this
/// convolution's slot in the layer's shared state cache: a layer runs several of them and
/// they share one cache object.
class InklingShortConvolution: Module {
    @ModuleInfo(key: "conv") var convolution: Conv1d

    let kernelSize: Int
    let convolutionIndex: Int

    init(channels: Int, kernelSize: Int, convolutionIndex: Int) {
        self.kernelSize = kernelSize
        self.convolutionIndex = convolutionIndex
        _convolution.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
            groups: channels, bias: false)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: ArraysCache? = nil,
        mask: MLXArray? = nil,
        residual: MLXArray? = nil
    ) -> MLXArray {
        let dtype = x.dtype
        let k = kernelSize

        // The fused path covers decode: a short block, no mask, the published tap count.
        if let cache, mask == nil, k == 4, x.dim(1) <= 8,
            Device.defaultDevice().deviceType == .gpu
        {
            let batch = x.dim(0)
            let length = x.dim(1)
            let channels = x.dim(2)
            let state =
                cache[convolutionIndex]
                ?? MLXArray.zeros([batch, k - 1, channels], dtype: .float32)

            let out = InklingShortConvolutionKernel.shared.kernel(
                [x, state, convolution.weight.reshaped(-1), residual ?? x],
                template: [
                    ("T", dtype),
                    ("B", batch),
                    ("L", length),
                    ("C", channels),
                    ("K", k),
                    ("HAS_RES", residual != nil ? 1 : 0),
                ],
                grid: ((channels + 31) / 32 * 32, batch, 1),
                threadGroup: (32, 1, 1),
                outputShapes: [[batch, length, channels], [batch, k - 1, channels]],
                outputDTypes: [dtype, .float32]
            )
            cache[convolutionIndex] = out[1]
            return out[0]
        }

        var xf = x.asType(.float32)
        let carried = xf
        // Padding must not enter the convolution, so it is zeroed before the window.
        if let mask {
            xf = MLX.where(mask[.ellipsis, .newAxis], xf, MLXArray(0, dtype: .float32))
        }

        let padded: MLXArray
        if let cache {
            let state =
                cache[convolutionIndex]
                ?? MLXArray.zeros([xf.dim(0), k - 1, xf.dim(-1)], dtype: xf.dtype)
            padded = concatenated([state, xf], axis: 1)
            cache[convolutionIndex] = nextInklingConvolutionState(
                state: state, inputs: xf, mask: mask)
        } else {
            padded = MLX.padded(
                xf, widths: [.init((0, 0)), .init((k - 1, 0)), .init((0, 0))])
        }

        var out = convolution(padded.asType(convolution.weight.dtype)).asType(.float32)
        out = (out + carried).asType(dtype)
        return residual.map { $0 + out } ?? out
    }
}

// MARK: - Configuration

/// Inkling's text stack.
///
/// Its layers alternate between sliding-window and full attention, and the two have
/// *different geometry* -- head count, KV head count and head dimension are all configured
/// separately for the sliding kind, which is why the attention reads them per layer rather
/// than once.
public struct InklingTextConfiguration: Decodable, Sendable {
    public var modelType: String = "inkling"
    public var hiddenSize: Int = 6144
    public var hiddenLayers: Int = 66
    public var vocabularySize: Int = 201_024
    public var unpaddedVocabularySize: Int?
    public var rmsNormEps: Float = 1e-6
    public var tieWordEmbeddings: Bool = false
    public var useEmbedNorm: Bool = true
    public var logitsMupWidthMultiplier: Float = 1.0
    public var maxPositionEmbeddings: Int = 1_048_576
    public var attentionHeads: Int = 64
    public var kvHeads: Int = 8
    public var headDim: Int = 128
    public var slidingAttentionHeads: Int = 64
    public var slidingKVHeads: Int = 16
    public var slidingHeadDim: Int = 128
    public var slidingWindowSize: Int = 512
    public var localLayerIDs: [Int]?
    public var layerTypes: [String]?
    public var relDimensions: Int = 16
    public var relativeExtent: Int = 1024
    public var logScalingFloor: Int?
    public var logScalingAlpha: Float = 0.1
    public var convolutionKernelSize: Int = 4
    public var denseMLPIndex: Int = 0
    public var mlpLayerTypes: [String]?
    /// Inkling names the MoE width `intermediate_size` and the dense one separately.
    public var intermediateSize: Int = 24576
    public var denseIntermediateSize: Int?
    public var routedExperts: Int = 256
    public var expertsPerToken: Int = 6
    public var sharedExperts: Int = 2
    public var routeScale: Float = 8.0

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case vocabularySize = "vocab_size"
        case unpaddedVocabularySize = "unpadded_vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case tieWordEmbeddings = "tie_word_embeddings"
        case useEmbedNorm = "use_embed_norm"
        case logitsMupWidthMultiplier = "logits_mup_width_multiplier"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case slidingAttentionHeads = "swa_num_attention_heads"
        case slidingKVHeads = "swa_num_key_value_heads"
        case slidingHeadDim = "swa_head_dim"
        case slidingWindowSize = "sliding_window_size"
        case localLayerIDs = "local_layer_ids"
        case layerTypes = "layer_types"
        case relDimensions = "d_rel"
        case relativeExtent = "rel_extent"
        case logScalingFloor = "log_scaling_n_floor"
        case logScalingAlpha = "log_scaling_alpha"
        case convolutionKernelSize = "sconv_kernel_size"
        case denseMLPIndex = "dense_mlp_idx"
        case mlpLayerTypes = "mlp_layer_types"
        case intermediateSize = "intermediate_size"
        case denseIntermediateSize = "dense_intermediate_size"
        case routedExperts = "n_routed_experts"
        case expertsPerToken = "num_experts_per_tok"
        case sharedExperts = "n_shared_experts"
        case routeScale = "route_scale"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func take<T: Decodable>(_ key: CodingKeys, _ fallback: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: key) ?? fallback
        }
        modelType = try take(.modelType, modelType)
        hiddenSize = try take(.hiddenSize, hiddenSize)
        hiddenLayers = try take(.hiddenLayers, hiddenLayers)
        vocabularySize = try take(.vocabularySize, vocabularySize)
        unpaddedVocabularySize = try c.decodeIfPresent(
            Int.self, forKey: .unpaddedVocabularySize)
        rmsNormEps = try take(.rmsNormEps, rmsNormEps)
        tieWordEmbeddings = try take(.tieWordEmbeddings, tieWordEmbeddings)
        useEmbedNorm = try take(.useEmbedNorm, useEmbedNorm)
        logitsMupWidthMultiplier = try take(
            .logitsMupWidthMultiplier, logitsMupWidthMultiplier)
        maxPositionEmbeddings = try take(.maxPositionEmbeddings, maxPositionEmbeddings)
        attentionHeads = try take(.attentionHeads, attentionHeads)
        kvHeads = try take(.kvHeads, kvHeads)
        headDim = try take(.headDim, headDim)
        slidingAttentionHeads = try take(.slidingAttentionHeads, slidingAttentionHeads)
        slidingKVHeads = try take(.slidingKVHeads, slidingKVHeads)
        slidingHeadDim = try take(.slidingHeadDim, slidingHeadDim)
        slidingWindowSize = try take(.slidingWindowSize, slidingWindowSize)
        localLayerIDs = try c.decodeIfPresent([Int].self, forKey: .localLayerIDs)
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes)
        relDimensions = try take(.relDimensions, relDimensions)
        relativeExtent = try take(.relativeExtent, relativeExtent)
        logScalingFloor = try c.decodeIfPresent(Int.self, forKey: .logScalingFloor)
        logScalingAlpha = try take(.logScalingAlpha, logScalingAlpha)
        convolutionKernelSize = try take(.convolutionKernelSize, convolutionKernelSize)
        denseMLPIndex = try take(.denseMLPIndex, denseMLPIndex)
        mlpLayerTypes = try c.decodeIfPresent([String].self, forKey: .mlpLayerTypes)
        intermediateSize = try take(.intermediateSize, intermediateSize)
        denseIntermediateSize = try c.decodeIfPresent(
            Int.self, forKey: .denseIntermediateSize)
        routedExperts = try take(.routedExperts, routedExperts)
        expertsPerToken = try take(.expertsPerToken, expertsPerToken)
        sharedExperts = try take(.sharedExperts, sharedExperts)
        routeScale = try take(.routeScale, routeScale)
    }

    /// Sliding-window (local) against full attention.
    ///
    /// Published checkpoints set one of two keys -- `layer_types` on the 0.6B,
    /// `local_layer_ids` on the 975B -- and the modulo is the reference default, which makes
    /// every sixth layer global.
    public func layerIsSliding(_ index: Int) -> Bool {
        if let layerTypes { return layerTypes[index] == "hybrid_sliding" }
        if let localLayerIDs { return Set(localLayerIDs).contains(index) }
        return (index + 1) % 6 != 0
    }

    public func layerIsDense(_ index: Int) -> Bool {
        if let mlpLayerTypes { return mlpLayerTypes[index] == "dense" }
        return index < denseMLPIndex
    }
}

// MARK: - Attention

/// Inkling's attention.
///
/// Three things set it apart from an ordinary block. Its q/k/v/r projections share one
/// stacked matrix, so decode does one matmul instead of four. Keys and values pass through
/// short convolutions before they are normed, which is where the layer's carried state
/// lives. And position enters through the additive banded mask rather than a rotary
/// embedding, from the `r` rows of that same projection.
class InklingAttention: Module {
    @ModuleInfo(key: "qkvr_proj") var qkvrProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "k_sconv") var keyConvolution: InklingShortConvolution
    @ModuleInfo(key: "v_sconv") var valueConvolution: InklingShortConvolution
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    @ParameterInfo(key: "rel_proj") var relProjection: MLXArray

    let isSliding: Bool
    let headDim: Int
    let heads: Int
    let kvHeads: Int
    let slidingWindow: Int
    let relativeExtent: Int
    let relDimensions: Int
    /// Not the usual inverse square root: Inkling scales by `1 / head_dim`.
    let scale: Float
    let logScalingFloor: Int?
    let logScalingAlpha: Float

    /// Widths of the four stacked projections, in order.
    let projectionWidths: (query: Int, key: Int, value: Int, rel: Int)

    init(_ config: InklingTextConfiguration, layerIndex: Int) {
        isSliding = config.layerIsSliding(layerIndex)
        headDim = isSliding ? config.slidingHeadDim : config.headDim
        heads = isSliding ? config.slidingAttentionHeads : config.attentionHeads
        kvHeads = isSliding ? config.slidingKVHeads : config.kvHeads
        slidingWindow = isSliding ? config.slidingWindowSize : 0
        relativeExtent = isSliding ? config.slidingWindowSize : config.relativeExtent
        relDimensions = config.relDimensions
        scale = 1.0 / Float(headDim)
        // Log scaling applies to the global layers only.
        logScalingFloor = isSliding ? nil : config.logScalingFloor
        logScalingAlpha = config.logScalingAlpha

        projectionWidths = (
            query: heads * headDim,
            key: kvHeads * headDim,
            value: kvHeads * headDim,
            rel: heads * relDimensions
        )
        let total =
            projectionWidths.query + projectionWidths.key + projectionWidths.value
            + projectionWidths.rel
        _qkvrProj.wrappedValue = Linear(config.hiddenSize, total, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        _keyConvolution.wrappedValue = InklingShortConvolution(
            channels: projectionWidths.key, kernelSize: config.convolutionKernelSize,
            convolutionIndex: 0)
        _valueConvolution.wrappedValue = InklingShortConvolution(
            channels: projectionWidths.value, kernelSize: config.convolutionKernelSize,
            convolutionIndex: 1)
        _queryNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _keyNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _relProjection.wrappedValue = MLXArray.zeros([relDimensions, relativeExtent])
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, cache: CacheList?, convolutionMask: MLXArray? = nil
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let kvCache = cache?[0]
        let convolutionCache =
            (cache?.children.count ?? 0) > 1
            ? cache?[1] as? ArraysCache : nil

        let projected = qkvrProj(x)
        let (dq, dk, dv, _) = projectionWidths

        var keys = keyConvolution(
            projected[.ellipsis, dq ..< (dq + dk)], cache: convolutionCache,
            mask: convolutionMask)
        var values = valueConvolution(
            projected[.ellipsis, (dq + dk) ..< (dq + dk + dv)], cache: convolutionCache,
            mask: convolutionMask)

        keys = keyNorm(keys.reshaped(batch, length, kvHeads, headDim))
            .transposed(0, 2, 1, 3)
        values = values.reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)

        if let kvCache {
            (keys, values) = kvCache.update(keys: keys, values: values)
        }
        var keyLength = keys.dim(2)

        var queries = queryNorm(
            projected[.ellipsis, ..<dq].reshaped(batch, length, heads, headDim)
        )
        .transposed(0, 2, 1, 3)
        let rel = projected[.ellipsis, (dq + dk + dv)...]
            .reshaped(batch, length, heads, relDimensions)

        // No query row can reach keys older than its window, so they are sliced off before
        // the mask and the attention rather than masked to -1e30 and multiplied anyway.
        if slidingWindow > 0, keyLength > length + slidingWindow - 1 {
            let start = keyLength - length - (slidingWindow - 1)
            keys = keys[0..., 0..., start..., 0...]
            values = values[0..., 0..., start..., 0...]
            keyLength = keys.dim(2)
        }
        let offset = keyLength - length

        var mask = bandedAdditiveMask(
            rel: rel,
            projection: relProjection.asType(x.dtype),
            queryOffset: offset,
            keyLength: keyLength,
            slidingWindow: slidingWindow,
            relativeExtent: relativeExtent,
            shapeReference: keys)

        // Global layers stretch the logit scale with absolute position, which has to reach
        // the mask's bias too or the two would no longer be on the same scale.
        if let logScalingFloor {
            let positions =
                (MLXArray(Int32(0) ..< Int32(length)) + Int32(offset + 1)).asType(.float32)
            var tau =
                1.0
                + logScalingAlpha
                * MLX.log(maximum(positions / Float(logScalingFloor), 1.0))
            tau = tau.reshaped(1, 1, length, 1).asType(x.dtype)
            queries = queries * tau
            mask = MLX.where(mask .> -1e29, mask * tau, mask)
        }

        var out = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: .array(mask))
        out = out.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        return oProj(out)
    }
}

// MARK: - Feed-forward

/// Dense SwiGLU with a learned output scale.
class InklingDenseMLP: SwiGLUMLP {
    @ParameterInfo(key: "global_scale") var globalScale: MLXArray

    init(_ config: InklingTextConfiguration) {
        _globalScale.wrappedValue = MLXArray.ones([1])
        super.init(
            inputDimensions: config.hiddenSize,
            hiddenDimensions: config.denseIntermediateSize ?? config.intermediateSize)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        super.callAsFunction(x) * globalScale
    }
}

/// Switched experts with per-expert scales.
///
/// NVFP4 checkpoints carry a scale on the gate and on the output; every other checkpoint
/// carries ones. The all-ones case is the common one and skips both gathers entirely, which
/// is why it is checked once and remembered rather than tested per token.
class InklingSwitchGLU: SwitchGLU {
    @ParameterInfo(key: "gate_scale") var gateScale: MLXArray
    @ParameterInfo(key: "out_scale") var outScale: MLXArray

    private var scalesAreTrivial: Bool?

    init(inputDimensions: Int, hiddenDimensions: Int, experts: Int) {
        _gateScale.wrappedValue = MLXArray.ones([experts])
        _outScale.wrappedValue = MLXArray.ones([experts])
        super.init(
            inputDims: inputDimensions, hiddenDims: hiddenDimensions, numExperts: experts)
    }

    /// Checked once, after load: a scale array is not going to change afterwards.
    func trivialScales() -> Bool {
        if let scalesAreTrivial { return scalesAreTrivial }
        let trivial =
            (gateScale .== 1).all().item(Bool.self)
            && (outScale .== 1).all().item(Bool.self)
        scalesAreTrivial = trivial
        return trivial
    }

    private func perExpert(_ scale: MLXArray, _ indices: MLXArray, like: MLXArray) -> MLXArray {
        var s = scale[indices].asType(like.dtype)
        while s.ndim < like.ndim {
            s = s[.ellipsis, .newAxis]
        }
        return s
    }

    override func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        if trivialScales() {
            return super.callAsFunction(x, indices)
        }
        // The scaled path drives the projections directly. Upstream additionally sorts by
        // expert for large token counts, which changes only the order work is issued in.
        let expanded = MLX.expandedDimensions(x, axes: [-2, -3])
        let up = upProj(expanded, indices)
        var gate = gateProj(expanded, indices)
        gate = gate * perExpert(gateScale, indices, like: gate)
        var out = downProj(compiledSiluProduct(gate, up), indices)
        out = out * perExpert(outScale, indices, like: out)
        return MLX.squeezed(out, axis: -2)
    }
}

/// The always-on shared experts, as one dense SwiGLU.
///
/// Their weights are concatenated at load (see ``sharedExpertsToDense(weights:)``) so the
/// fixed-index gather becomes three plain matmuls. The per-token expert weights arrive
/// already broadcast over the expert-major intermediate and are applied before the down
/// projection, which distributes over the concatenated experts exactly.
class InklingSharedExpertsDense: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(inputDimensions: Int, hiddenDimensions: Int, experts: Int) {
        _gateProj.wrappedValue = Linear(
            inputDimensions, experts * hiddenDimensions, bias: false)
        _upProj.wrappedValue = Linear(
            inputDimensions, experts * hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(
            experts * hiddenDimensions, inputDimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gamma: MLXArray) -> MLXArray {
        downProj(compiledSiluProduct(gateProj(x), upProj(x)) * gamma)
    }
}

/// Sigmoid-gated fine-grained MoE.
///
/// Two things are unusual. Selection and weighting read different quantities: experts are
/// chosen on `sigmoid(logit) + correction bias`, but the weights come from a softmax over
/// the *log-sigmoid* of the raw logits. And the always-on shared experts enter that softmax
/// alongside the routed ones, so they compete for the same weight mass rather than being
/// added on top at full strength.
class InklingSparseMoE: Module, UnaryLayer {
    @ParameterInfo(key: "gate_weight") var gateWeight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray
    @ParameterInfo(key: "global_scale") var globalScale: MLXArray
    @ModuleInfo(key: "switch_mlp") var switchMLP: InklingSwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: InklingSharedExpertsDense

    let routedExperts: Int
    let sharedExpertCount: Int
    let topK: Int
    let routeScale: Float
    let intermediateSize: Int

    init(_ config: InklingTextConfiguration) {
        routedExperts = config.routedExperts
        sharedExpertCount = config.sharedExperts
        topK = config.expertsPerToken
        routeScale = config.routeScale
        intermediateSize = config.intermediateSize

        // One gate covers both kinds: the shared experts' logits are the trailing rows.
        _gateWeight.wrappedValue = MLXArray.zeros([
            routedExperts + sharedExpertCount, config.hiddenSize,
        ])
        _correctionBias.wrappedValue = MLXArray.zeros([routedExperts])
        _globalScale.wrappedValue = MLXArray.ones([1])
        _switchMLP.wrappedValue = InklingSwitchGLU(
            inputDimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize,
            experts: routedExperts)
        _sharedExperts.wrappedValue = InklingSharedExpertsDense(
            inputDimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize,
            experts: sharedExpertCount)
        super.init()
    }

    /// Expert selection and routing weights.
    ///
    /// - Returns: the chosen expert indices, their weights, and the shared experts' weights
    ///   already broadcast over the expert-major intermediate.
    func route(
        _ logits: MLXArray
    ) -> (indices: MLXArray, weights: MLXArray, gamma: MLXArray) {
        // The correction bias steers selection only; it never reaches the weights.
        let scores = sigmoid(logits.asType(.float32))
        let biased = scores[0..., ..<routedExperts] + correctionBias
        let indices = argPartition(-biased, kth: topK - 1, axis: -1)[0..., ..<topK]

        let routedLogits = logits[0..., ..<routedExperts]
        let sharedLogits = logits[0..., (-sharedExpertCount)...]
        let selected = concatenated(
            [takeAlong(routedLogits, indices, axis: -1), sharedLogits], axis: -1
        )
        .asType(.float32)

        // log-sigmoid, then a softmax over routed and shared together.
        let logProbabilities = -logAddExp(MLXArray.zeros(like: selected), -selected)
        let weights =
            exp(logProbabilities - logSumExp(logProbabilities, axis: -1, keepDims: true))
            * routeScale * globalScale

        let topKWeights = weights[0..., ..<topK].asType(logits.dtype)
        let gamma = repeated(
            weights[0..., (-sharedExpertCount)...].asType(logits.dtype),
            count: intermediateSize, axis: -1)
        return (indices.asType(.uint32), topKWeights, gamma)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let width = x.dim(2)
        let flat = x.reshaped(-1, width)

        let logits = flat.matmul(gateWeight.asType(x.dtype).transposed())
        let (indices, weights, gamma) = route(logits)

        let routed = (switchMLP(flat, indices) * weights[.ellipsis, .newAxis]).sum(axis: -2)
        let shared = sharedExperts(flat, gamma: gamma)
        return (routed + shared).reshaped(batch, length, width).asType(x.dtype)
    }
}

// MARK: - Load-time fusions

/// Stacks each layer's separate q/k/v/r projections into the single `qkvr_proj`.
///
/// Row-concatenating quantized matrices is exact: every output row keeps its own
/// quantization groups, so the constituents' settings carry over unchanged.
func fuseInklingQKVR(weights: [String: MLXArray]) throws -> [String: MLXArray] {
    var out = weights
    let prefixes = Set(
        weights.keys.filter { $0.hasSuffix(".self_attn.q_proj.weight") }
            .map { String($0.dropLast("q_proj.weight".count)) })

    for prefix in prefixes {
        for leaf in ["weight", "scales", "biases"] {
            let parts = ["q", "k", "v", "r"].map {
                out.removeValue(forKey: "\(prefix)\($0)_proj.\(leaf)")
            }
            if parts.allSatisfy({ $0 != nil }) {
                out["\(prefix)qkvr_proj.\(leaf)"] = concatenated(parts.map { $0! }, axis: 0)
            } else if parts.contains(where: { $0 != nil }) {
                throw InklingWeightError.partialProjectionSet(prefix: prefix, leaf: leaf)
            }
        }
    }
    return out
}

/// Remaps switched-shaped shared-expert tensors onto the dense concatenated layout.
///
/// Expert-major on the intermediate axis: gate and up stack experts along rows, and down
/// stacks them along input columns.
func inklingSharedExpertsToDense(weights: [String: MLXArray]) -> [String: MLXArray] {
    var out: [String: MLXArray] = [:]
    for (key, value) in weights {
        guard key.contains(".shared_experts."), value.ndim == 3 else {
            out[key] = value
            continue
        }
        if key.contains(".down_proj.") {
            out[key] = value.transposed(1, 0, 2).reshaped(value.dim(1), -1)
        } else {
            out[key] = value.reshaped(-1, value.dim(2))
        }
    }
    return out
}

public enum InklingWeightError: Error, CustomStringConvertible {
    case partialProjectionSet(prefix: String, leaf: String)

    public var description: String {
        switch self {
        case .partialProjectionSet(let prefix, let leaf):
            "Inkling found a partial q/k/v/r \(leaf) set under \(prefix); the stacked "
                + "qkvr_proj needs all four or none."
        }
    }
}

// MARK: - Layer and model

/// Builds the convolution mask for a batch whose rows have different valid spans.
///
/// A row's padding must not enter its convolution state, and a row that has already ended
/// must not keep extending it. Nil for a single unpadded stream, which is the common case.
func inklingConvolutionPaddingMask(cache: ArraysCache?, length: Int) -> MLXArray? {
    guard let cache else { return nil }
    let positions = MLXArray(Int32(0) ..< Int32(length))[.newAxis, 0...]
    var mask: MLXArray?
    if let leftPadding = cache.leftPadding {
        mask = positions .>= leftPadding[0..., .newAxis]
    }
    if let lengths = cache.lengths {
        let within = positions .< lengths[0..., .newAxis]
        mask = mask.map { $0 .&& within } ?? within
    }
    return mask
}

/// One Inkling block.
///
/// Four short convolutions run per layer, not two: the attention holds one on its keys and
/// one on its values, and the layer holds one after attention and one after the MLP. They
/// share a single state cache and are told apart by slot, which is why the cache is sized
/// four.
class InklingDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: InklingAttention
    @ModuleInfo var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attn_sconv") var attentionConvolution: InklingShortConvolution
    @ModuleInfo(key: "mlp_sconv") var mlpConvolution: InklingShortConvolution

    init(_ config: InklingTextConfiguration, layerIndex: Int) {
        _attention.wrappedValue = InklingAttention(config, layerIndex: layerIndex)
        let feedForward: UnaryLayer =
            config.layerIsDense(layerIndex)
            ? InklingDenseMLP(config) : InklingSparseMoE(config)
        _mlp.wrappedValue = feedForward
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attentionConvolution.wrappedValue = InklingShortConvolution(
            channels: config.hiddenSize, kernelSize: config.convolutionKernelSize,
            convolutionIndex: 2)
        _mlpConvolution.wrappedValue = InklingShortConvolution(
            channels: config.hiddenSize, kernelSize: config.convolutionKernelSize,
            convolutionIndex: 3)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, cache: CacheList?, convolutionMask: MLXArray? = nil
    ) -> MLXArray {
        let convolutionCache =
            (cache?.children.count ?? 0) > 1
            ? cache?[1] as? ArraysCache : nil
        let mask =
            convolutionMask
            ?? inklingConvolutionPaddingMask(cache: convolutionCache, length: x.dim(1))

        // The convolutions carry the residual through rather than adding it separately,
        // which is what their fused kernel is written against.
        let attended = attention(inputLayerNorm(x), cache: cache, convolutionMask: mask)
        let h = attentionConvolution(
            attended, cache: convolutionCache, mask: mask, residual: x)
        let projected = mlp(postAttentionLayerNorm(h))
        let out = mlpConvolution(
            projected, cache: convolutionCache, mask: mask, residual: h)
        convolutionCache?.advance(x.dim(1))
        return out
    }
}

/// Inkling's text stack.
class InklingModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_norm") var embedNorm: RMSNorm?
    @ModuleInfo var layers: [InklingDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    let config: InklingTextConfiguration

    init(_ config: InklingTextConfiguration) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        if config.useEmbedNorm {
            _embedNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            InklingDecoderLayer(config, layerIndex: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func embed(_ inputs: MLXArray) -> MLXArray {
        let h = embedTokens(inputs)
        return embedNorm.map { $0(h) } ?? h
    }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        inputEmbeddings: MLXArray? = nil,
        skipFinalNorm: Bool = false
    ) -> MLXArray {
        var h = inputEmbeddings ?? embed(inputs)
        for (index, layer) in layers.enumerated() {
            h = layer(h, cache: cache?[index] as? CacheList)
        }
        return skipFinalNorm ? h : norm(h)
    }
}

/// Inkling as a text model.
public final class InklingModel: Module, LLMModel, KVCacheDimensionProvider {
    @ModuleInfo var model: InklingModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let config: InklingTextConfiguration
    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] {
        (0 ..< config.hiddenLayers).map {
            config.layerIsSliding($0) ? config.slidingKVHeads : config.kvHeads
        }
    }

    public init(_ config: InklingTextConfiguration) {
        self.config = config
        _model.wrappedValue = InklingModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(from: model(inputs, cache: cache))
    }

    /// The width multiplier divides the hidden state before the head, and the padded rows
    /// of the vocabulary are cut off afterwards -- a checkpoint's embedding table is padded
    /// to a convenient width, but those rows are not tokens.
    public func logits(from hidden: MLXArray) -> MLXArray {
        let scaled = hidden / config.logitsMupWidthMultiplier
        var out = lmHead.map { $0(scaled) } ?? model.embedTokens.asLinear(scaled)
        if let unpadded = config.unpaddedVocabularySize, unpadded < out.dim(-1) {
            out = out[.ellipsis, ..<unpadded]
        }
        return out
    }

    public var loraLayers: [Module] { model.layers }

    /// Per layer: a KV cache and the four convolution slots that layer's convolutions share.
    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        (0 ..< config.hiddenLayers).map { _ in
            CacheList(MaterializedKVCache(KVCacheSimple()), ArraysCache(size: 4))
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let dense = inklingSharedExpertsToDense(weights: weights)
        return (try? fuseInklingQKVR(weights: dense)) ?? dense
    }
}
