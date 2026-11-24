import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct Glm5NextConfiguration: Decodable, Sendable {

    /// Everything the MoE and the latent attention need is DeepSeek V3.2's, so the
    /// configuration carries that one and adds the schedules and the linear half on top.
    public struct TextConfiguration: Decodable, Sendable {
        public var deepseek: DeepseekV32Configuration
        public var layerTypes: [String]
        public var mlpLayerTypes: [String]

        public var linearHeads: Int = 64
        public var linearHeadDim: Int = 128
        public var linearConvKernelDim: Int = 4
        public var linearLowerBound: Float? = -5.0

        public var hcMultiplier: Int = 4
        public var hcEps: Float = 1e-6
        public var hcSinkhornIterations: Int = 20

        public var indexKPool: Int = 4
        public var indexKPoolAlwaysSelectTail: Bool = true
        public var mlaUseNope: Bool = true
        public var tieWordEmbeddings: Bool = false

        enum CodingKeys: String, CodingKey {
            case layerTypes = "layer_types"
            case mlpLayerTypes = "mlp_layer_types"
            case linearAttentionConfig = "linear_attn_config"
            case linearHeads = "linear_num_heads"
            case linearHeadDim = "linear_head_dim"
            case linearConvKernelDim = "linear_conv_kernel_dim"
            case linearLowerBound = "linear_lower_bound"
            case hcMultiplier = "hc_mult"
            case hcEps = "hc_eps"
            case hcSinkhornIterations = "hc_sinkhorn_iters"
            case indexKPool = "index_kpool"
            case indexKPoolAlwaysSelectTail = "index_kpool_always_select_tail"
            case mlaUseNope = "mla_use_nope"
            case tieWordEmbeddings = "tie_word_embeddings"
        }

        /// The nested `linear_attn_config` object, whose keys have their own names.
        private struct LinearAttentionConfiguration: Decodable {
            var heads: Int?
            var headDim: Int?
            var convKernel: Int?
            var safeGate: Bool?

            /// A present-but-null bound is not the same as an absent one: it removes the
            /// bound, where an absent key leaves whatever the flat keys said.
            var declaresGateLowerBound = false
            var gateLowerBound: Float?

            enum CodingKeys: String, CodingKey {
                case heads = "num_heads"
                case headDim = "head_dim"
                case convKernel = "short_conv_kernel_size"
                case gateLowerBound = "gate_lower_bound"
                case safeGate = "safe_gate"
            }

            init(from decoder: Decoder) throws {
                let c = try decoder.container(keyedBy: CodingKeys.self)
                heads = try c.decodeIfPresent(Int.self, forKey: .heads)
                headDim = try c.decodeIfPresent(Int.self, forKey: .headDim)
                convKernel = try c.decodeIfPresent(Int.self, forKey: .convKernel)
                safeGate = try c.decodeIfPresent(Bool.self, forKey: .safeGate)
                declaresGateLowerBound = c.contains(.gateLowerBound)
                gateLowerBound = try c.decodeIfPresent(Float.self, forKey: .gateLowerBound)
            }
        }

        public init(from decoder: Decoder) throws {
            deepseek = try DeepseekV32Configuration(from: decoder)
            let c = try decoder.container(keyedBy: CodingKeys.self)

            let layers = deepseek.hiddenLayers
            layerTypes =
                try c.decodeIfPresent([String].self, forKey: .layerTypes)
                ?? Array(repeating: "full_attention", count: layers)
            mlpLayerTypes =
                try c.decodeIfPresent([String].self, forKey: .mlpLayerTypes)
                ?? Array(repeating: "sparse", count: layers)

            linearHeads = try c.decodeIfPresent(Int.self, forKey: .linearHeads) ?? linearHeads
            linearHeadDim =
                try c.decodeIfPresent(Int.self, forKey: .linearHeadDim) ?? linearHeadDim
            linearConvKernelDim =
                try c.decodeIfPresent(Int.self, forKey: .linearConvKernelDim)
                ?? linearConvKernelDim
            if c.contains(.linearLowerBound) {
                linearLowerBound = try c.decodeIfPresent(
                    Float.self, forKey: .linearLowerBound)
            }

            hcMultiplier = try c.decodeIfPresent(Int.self, forKey: .hcMultiplier) ?? hcMultiplier
            hcEps = try c.decodeIfPresent(Float.self, forKey: .hcEps) ?? hcEps
            hcSinkhornIterations =
                try c.decodeIfPresent(Int.self, forKey: .hcSinkhornIterations)
                ?? hcSinkhornIterations

            indexKPool = try c.decodeIfPresent(Int.self, forKey: .indexKPool) ?? indexKPool
            indexKPoolAlwaysSelectTail =
                try c.decodeIfPresent(Bool.self, forKey: .indexKPoolAlwaysSelectTail)
                ?? indexKPoolAlwaysSelectTail
            mlaUseNope = try c.decodeIfPresent(Bool.self, forKey: .mlaUseNope) ?? mlaUseNope
            tieWordEmbeddings =
                try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
                ?? tieWordEmbeddings

            // The nested object wins over the flat keys, and a safe gate with no explicit
            // bound still gets one.
            if let nested = try c.decodeIfPresent(
                LinearAttentionConfiguration.self, forKey: .linearAttentionConfig)
            {
                linearHeads = nested.heads ?? linearHeads
                linearHeadDim = nested.headDim ?? linearHeadDim
                linearConvKernelDim = nested.convKernel ?? linearConvKernelDim
                if nested.declaresGateLowerBound {
                    linearLowerBound = nested.gateLowerBound
                }
                if nested.safeGate ?? true, linearLowerBound == nil {
                    linearLowerBound = -5.0
                }
            }
        }

        public func isLinear(layerIndex: Int) -> Bool {
            layerIndex < layerTypes.count && layerTypes[layerIndex] == "linear_attention"
        }

        public func isSparseMLP(layerIndex: Int) -> Bool {
            guard deepseek.routedExperts != nil,
                layerIndex >= deepseek.firstKDenseReplace,
                layerIndex < mlpLayerTypes.count
            else {
                return false
            }
            return mlpLayerTypes[layerIndex] == "sparse"
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public var modelType: String = "glm5_next_vision"
        public var depth: Int = 24
        public var hiddenSize: Int = 1536
        public var intermediateSize: Int = 4096
        public var heads: Int = 12
        public var patchSize: Int = 14
        public var outHiddenSize: Int = 4096
        public var projectionIntermediateSize: Int = 10240
        public var imageSize: Int = 448
        public var channels: Int = 3
        public var rmsNormEps: Float = 1e-5
        public var attentionBias: Bool = true
        public var spatialMergeSize: Int = 2
        public var temporalPatchSize: Int = 2
        public var swiGLULimit: Float = 10.0

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case depth
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case heads = "num_heads"
            case patchSize = "patch_size"
            case outHiddenSize = "out_hidden_size"
            case projectionIntermediateSize = "projection_intermediate_size"
            case imageSize = "image_size"
            case channels = "in_channels"
            case rmsNormEps = "rms_norm_eps"
            case attentionBias = "attention_bias"
            case spatialMergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
            case swiGLULimit = "swiglu_limit"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
            depth = try c.decodeIfPresent(Int.self, forKey: .depth) ?? depth
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? intermediateSize
            heads = try c.decodeIfPresent(Int.self, forKey: .heads) ?? heads
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? patchSize
            outHiddenSize =
                try c.decodeIfPresent(Int.self, forKey: .outHiddenSize) ?? outHiddenSize
            projectionIntermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .projectionIntermediateSize)
                ?? projectionIntermediateSize
            imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? imageSize
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? channels
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? rmsNormEps
            attentionBias =
                try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? attentionBias
            spatialMergeSize =
                try c.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? spatialMergeSize
            temporalPatchSize =
                try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize)
                ?? temporalPatchSize
            swiGLULimit =
                try c.decodeIfPresent(Float.self, forKey: .swiGLULimit) ?? swiGLULimit
        }
    }

    public var modelType: String
    public var textConfiguration: TextConfiguration
    public var visionConfiguration: VisionConfiguration
    public var imageTokenID: Int
    public var videoTokenID: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case imageTokenID = "image_token_id"
        case videoTokenID = "video_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "glm5_next"
        textConfiguration = try c.decode(TextConfiguration.self, forKey: .textConfiguration)
        visionConfiguration =
            try c.decodeIfPresent(VisionConfiguration.self, forKey: .visionConfiguration)
            ?? VisionConfiguration(from: decoder)
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? 154_854
        videoTokenID = try c.decodeIfPresent(Int.self, forKey: .videoTokenID) ?? 154_855
    }
}

// MARK: - Linear attention

/// RMS normalization with a sigmoid gate applied after the scale, in float32 throughout.
class Glm5NextRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray) -> MLXArray {
        let dtype = hiddenStates.dtype
        var x = hiddenStates.asType(.float32)
        let variance = (x * x).mean(axis: -1, keepDims: true)
        x = x * rsqrt(variance + eps)
        x = weight.asType(.float32) * x
        return (x * sigmoid(gate.asType(.float32))).asType(dtype)
    }
}

/// Holds the forget gate's parameters. The decay itself is computed inside the shared
/// gated-delta update, which is where the bound is applied.
class Glm5NextForgetGate: Module {
    @ModuleInfo(key: "f_a_proj") var fAProj: Linear
    @ModuleInfo(key: "f_b_proj") var fBProj: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray

    let lowerBound: Float?

    init(_ config: Glm5NextConfiguration.TextConfiguration) {
        let width = config.linearHeadDim * config.linearHeads
        _fAProj.wrappedValue = Linear(
            config.deepseek.hiddenSize, config.linearHeadDim, bias: false)
        _fBProj.wrappedValue = Linear(config.linearHeadDim, width, bias: false)
        _dtBias.wrappedValue = MLXArray.zeros([width])
        _aLog.wrappedValue = MLXArray.zeros([config.linearHeads])
        lowerBound = config.linearLowerBound
        super.init()
    }
}

private func l2Normalized(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    x * rsqrt((x * x).sum(axis: -1, keepDims: true) + eps)
}

/// Kimi Delta Attention: a short depthwise convolution over the fused q, k and v, then a
/// delta-rule recurrence whose decay is per key dimension rather than per head.
class Glm5NextLinearAttention: Module {
    let heads: Int
    let headDim: Int
    let qkvDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "conv1d") var conv: Conv1d
    @ModuleInfo(key: "forget_gate") var forgetGate: Glm5NextForgetGate
    @ModuleInfo(key: "b_proj") var bProj: Linear
    @ModuleInfo(key: "g_a_proj") var gAProj: Linear
    @ModuleInfo(key: "g_b_proj") var gBProj: Linear
    @ModuleInfo(key: "o_norm") var oNorm: Glm5NextRMSNormGated
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: Glm5NextConfiguration.TextConfiguration) {
        heads = config.linearHeads
        headDim = config.linearHeadDim
        qkvDim = heads * headDim
        convKernelSize = config.linearConvKernelDim
        convDim = qkvDim * 3

        let hidden = config.deepseek.hiddenSize
        _qProj.wrappedValue = Linear(hidden, qkvDim, bias: false)
        _kProj.wrappedValue = Linear(hidden, qkvDim, bias: false)
        _vProj.wrappedValue = Linear(hidden, qkvDim, bias: false)
        _conv.wrappedValue = Conv1d(
            inputChannels: convDim, outputChannels: convDim,
            kernelSize: convKernelSize, padding: 0, groups: convDim, bias: false)
        _forgetGate.wrappedValue = Glm5NextForgetGate(config)
        _bProj.wrappedValue = Linear(hidden, heads, bias: false)
        _gAProj.wrappedValue = Linear(hidden, headDim, bias: false)
        _gBProj.wrappedValue = Linear(headDim, qkvDim, bias: false)
        _oNorm.wrappedValue = Glm5NextRMSNormGated(
            dimensions: headDim, eps: config.deepseek.rmsNormEps)
        _oProj.wrappedValue = Linear(qkvDim, hidden, bias: false)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, mask: MLXArray?, cache: ArraysCache?)
        -> MLXArray
    {
        let (B, S) = (inputs.dim(0), inputs.dim(1))

        var mixed = concatenated([qProj(inputs), kProj(inputs), vProj(inputs)], axis: -1)
        if let mask {
            mixed = MLX.where(mask[.ellipsis, .newAxis], mixed, MLXArray(0))
        }

        let tail =
            cache?[0]
            ?? MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: inputs.dtype)
        let padded = concatenated([tail, mixed], axis: 1)
        cache?[0] = padded[0..., (1 - convKernelSize)...]
        let convolved = silu(conv(padded))

        let parts = split(convolved, indices: [qkvDim, 2 * qkvDim], axis: -1)
        var q = parts[0].reshaped(B, S, heads, headDim)
        var k = parts[1].reshaped(B, S, heads, headDim)
        let v = parts[2].reshaped(B, S, heads, headDim)

        let a = forgetGate.fBProj(forgetGate.fAProj(inputs))
            .reshaped(B, S, heads, headDim)
        let dtype = q.dtype
        q = (l2Normalized(q.asType(.float32)) * pow(Float(headDim), -0.5)).asType(dtype)
        k = l2Normalized(k.asType(.float32)).asType(dtype)

        let (out, state) = gatedDeltaUpdate(
            q: q, k: k, v: v, a: a, b: bProj(inputs),
            aLog: forgetGate.aLog.reshaped(heads, 1),
            dtBias: forgetGate.dtBias.reshaped(heads, headDim),
            state: cache?[1],
            mask: mask,
            lowerBound: forgetGate.lowerBound)
        if let cache {
            cache[1] = state
            cache.advance(S)
        }

        let gate = gBProj(gAProj(inputs)).reshaped(B, S, heads, headDim)
        return oProj(oNorm(out, gate: gate).reshaped(B, S, -1))
    }
}

// MARK: - Indexer

/// Chooses which cached positions the latent attention may read.
///
/// Keys are grouped into pools of `index_kpool` and scored one pool at a time, so the
/// selection is `index_topk / index_kpool` pools wide and expands back to whole pools.
/// The partial pool at the end of the visible range is appended separately, because it
/// would otherwise be invalid and never selected -- and it is the most recent context.
class Glm5NextIndexer: Module {
    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo var wk: Linear
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear

    @ParameterInfo(key: "index_kpool_compress_ape") var compressAPE: MLXArray
    @ParameterInfo(key: "index_kpool_compress_gate") var compressGate: MLXArray

    let heads: Int
    let headDim: Int
    let indexTopK: Int
    let kPool: Int
    let alwaysSelectTail: Bool
    let softmaxScale: Float

    /// Query positions are scored against every pool at once, so a one-shot prefill would
    /// materialize `[batch, length, heads, pools]`. Chunking bounds that.
    static let queryChunk = 512

    init(_ config: Glm5NextConfiguration.TextConfiguration) {
        let deepseek = config.deepseek
        heads = deepseek.indexHeads
        headDim = deepseek.indexHeadDim
        indexTopK = deepseek.indexTopK
        kPool = config.indexKPool
        alwaysSelectTail = config.indexKPoolAlwaysSelectTail
        softmaxScale = pow(Float(deepseek.indexHeadDim), -0.5)

        _wqB.wrappedValue = Linear(deepseek.qLoraRank, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(deepseek.hiddenSize, headDim, bias: false)
        _kNorm.wrappedValue = LayerNorm(dimensions: headDim)
        _weightsProj.wrappedValue = Linear(deepseek.hiddenSize, heads, bias: false)
        _compressAPE.wrappedValue = MLXArray.zeros([kPool, headDim])
        _compressGate.wrappedValue = MLXArray.zeros([headDim, deepseek.hiddenSize])
        super.init()
    }

    /// The first valid key in each row, or the row's length when none is.
    private func firstValid(_ valid: MLXArray, length: Int) -> MLXArray {
        let anyValid = any(valid, axis: -1)
        return MLX.where(anyValid, argMax(valid.asType(.int32), axis: -1), MLXArray(length))
    }

    private func pooledStates(keys: MLXArray, gates: MLXArray, valid: MLXArray) -> (
        keys: MLXArray, indices: MLXArray, valid: MLXArray
    ) {
        let (B, S, hd) = (keys.dim(0), keys.dim(1), keys.dim(2))
        let pools = (S + kPool - 1) / kPool
        let first = firstValid(valid, length: S)

        let offsets = MLXArray(0 ..< (pools * kPool)).reshaped(1, pools, kPool)
        var indices = first[0..., .newAxis, .newAxis] + offsets
        let safe = clip(indices, min: 0, max: S - 1)
        let flat = safe.reshaped(B, pools * kPool)
        let gathered = broadcast(flat[.ellipsis, .newAxis], to: [B, pools * kPool, hd])

        let groupedKeys = takeAlong(keys, gathered, axis: 1).reshaped(B, pools, kPool, hd)
        let groupedGates = takeAlong(gates, gathered, axis: 1).reshaped(B, pools, kPool, hd)
        var groupedValid =
            takeAlong(valid.asType(.int32), flat, axis: 1).reshaped(B, pools, kPool) .> 0
        groupedValid = groupedValid .&& (indices .< S)

        let poolValid = all(groupedValid, axis: -1)
        indices = MLX.where(groupedValid, indices, MLXArray(Int32(-1)))

        var logits = groupedGates + compressAPE[.newAxis, .newAxis]
        logits = MLX.where(
            groupedValid[.ellipsis, .newAxis], logits, MLXArray(Float(-1e30)))
        var probabilities = softmax(logits, axis: 2)
        probabilities = MLX.where(isNaN(probabilities), MLXArray(Float(0)), probabilities)

        return (
            keys: (probabilities * groupedKeys).sum(axis: 2),
            indices: indices,
            valid: poolValid
        )
    }

    /// The keys in the incomplete pool at the end of each query's visible range.
    func visibleTail(visible: MLXArray, valid: MLXArray) -> MLXArray {
        let (B, S, Kv) = (visible.dim(0), visible.dim(1), visible.dim(2))
        let width = kPool - 1
        let first = firstValid(valid, length: Kv)

        let visibleCount = visible.asType(.int32).sum(axis: -1)
        // Integer division: `/` on an integer array promotes to float, which would make
        // every tail empty.
        let tailCount = visibleCount - floorDivide(visibleCount, kPool) * kPool
        let offsets = MLXArray(0 ..< width)
        let start = first[0..., .newAxis] + visibleCount - tailCount
        var indices = start[.ellipsis, .newAxis] + offsets

        let inRange =
            (offsets[.newAxis, .newAxis, 0...] .< tailCount[.ellipsis, .newAxis])
            .&& (indices .< Kv)
        let clamped = clip(indices, min: 0, max: Kv - 1)
        let stillVisible = takeAlong(visible, clamped, axis: -1)
        indices = MLX.where(inRange .&& stillVisible, indices, MLXArray(Int32(-1)))
        _ = (B, S)
        return indices
    }

    /// `nil` when everything cached already fits inside the budget, which is what tells
    /// the attention to stay dense.
    func callAsFunction(
        _ x: MLXArray, queryRank: MLXArray, mask: MLXArray?, cache: KVCache?
    ) -> MLXArray? {
        let (B, S) = (x.dim(0), x.dim(1))
        let q = wqB(queryRank).reshaped(B, S, heads, headDim)
        let k = kNorm(wk(x)).reshaped(B, S, headDim)
        let gates = x.matmul(compressGate.swappedAxes(-1, -2))

        let currentValid: MLXArray
        if let mask, mask.dtype == .bool, mask.shape == [B, S] {
            currentValid = mask
        } else {
            currentValid = MLXArray.ones([B, S], dtype: .bool)
        }

        let packed = concatenated(
            [k, gates, currentValid.asType(k.dtype)[.ellipsis, .newAxis]], axis: -1)

        let packedFull: MLXArray
        if let cache {
            let (keys, _) = cache.update(
                keys: packed.expandedDimensions(axis: 1),
                values: MLXArray.zeros([B, 1, S, 0]))
            packedFull = keys[0..., 0]
        } else {
            packedFull = packed
        }

        let T = packedFull.dim(1)
        if T <= indexTopK { return nil }

        let split = MLX.split(packedFull, indices: [headDim, 2 * headDim], axis: -1)
        let keysFull = split[0]
        let gatesFull = split[1]
        let valid = split[2][.ellipsis, 0] .> 0

        let offset = T - S
        let keyPositions = MLXArray(0 ..< T)
        let pool = pooledStates(keys: keysFull, gates: gatesFull, valid: valid)
        let pools = pool.keys.dim(1)
        let selectCount = Swift.min(indexTopK / kPool, pools)
        let poolEnd = clip(pool.indices[.ellipsis, -1], min: 0, max: T - 1)
        let poolKeysTransposed = pool.keys.expandedDimensions(axis: 1).swappedAxes(-1, -2)

        let tailOn = alwaysSelectTail && kPool > 1
        let outputWidth = indexTopK + (tailOn ? kPool - 1 : 0)
        let chunk = S > Self.queryChunk ? Self.queryChunk : S

        var chunks: [MLXArray] = []
        var start = 0
        while start < S {
            let end = Swift.min(start + chunk, S)
            let width = end - start
            let queryPositions = MLXArray(Int32(offset + start) ..< Int32(offset + end))
            let visible =
                (keyPositions[.newAxis, .newAxis, 0...]
                    .<= queryPositions[.newAxis, 0..., .newAxis])
                .&& valid[0..., .newAxis, 0...]

            var scores = q[0..., start ..< end].matmul(poolKeysTransposed)
            scores = maximum(scores * softmaxScale, 0)
            let weights = weightsProj(x[0..., start ..< end]) * pow(Float(heads), -0.5)
            var indexScores = (weights[.ellipsis, .newAxis] * scores).sum(axis: 2)

            let poolVisible = takeAlong(
                visible, broadcast(poolEnd[0..., .newAxis, 0...], to: [B, width, pools]),
                axis: -1)
            let candidates = poolVisible .&& pool.valid.expandedDimensions(axis: 1)
            indexScores = MLX.where(candidates, indexScores, MLXArray(Float(-1e30)))

            let order = argSort(-indexScores, axis: -1)
            let selected = order[.ellipsis, ..<selectCount]
            let selectedValid = takeAlong(candidates, selected, axis: -1)

            let expandedIndices = broadcast(
                pool.indices.expandedDimensions(axis: 1), to: [B, width, pools, kPool])
            let expandedSelection = broadcast(
                selected[.ellipsis, .newAxis], to: [B, width, selectCount, kPool])
            var topK = takeAlong(expandedIndices, expandedSelection, axis: 2)
                .reshaped(B, width, selectCount * kPool)
            let keep = broadcast(
                selectedValid[.ellipsis, .newAxis], to: [B, width, selectCount, kPool]
            ).reshaped(B, width, selectCount * kPool)
            topK = MLX.where(keep, topK, MLXArray(Int32(-1)))

            if tailOn {
                topK = concatenated(
                    [topK, visibleTail(visible: visible, valid: valid)], axis: -1)
            }
            if topK.dim(-1) < outputWidth {
                topK = concatenated(
                    [
                        topK,
                        MLXArray.full(
                            [B, width, outputWidth - topK.dim(-1)], values: MLXArray(Int32(-1))),
                    ], axis: -1)
            }
            topK = topK[.ellipsis, ..<outputWidth]
            topK = MLX.where(
                currentValid[0..., start ..< end][.ellipsis, .newAxis], topK,
                MLXArray(Int32(-1)))

            chunks.append(topK)
            start = end
        }

        let topK = chunks.count == 1 ? chunks[0] : concatenated(chunks, axis: 1)
        return topK.expandedDimensions(axis: 1).asType(.int32)
    }
}

// MARK: - Sparse latent attention

/// Multi-head latent attention, NoPE by design, reading only the positions the indexer
/// selected.
class Glm5NextSparseAttention: Module {
    let heads: Int
    let qHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let scale: Float

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMQA: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: MultiLinear
    @ModuleInfo(key: "unembed_out") var unembedOut: MultiLinear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo var indexer: Glm5NextIndexer

    init(_ config: Glm5NextConfiguration.TextConfiguration) {
        let deepseek = config.deepseek
        precondition(
            config.mlaUseNope || deepseek.qkRopeHeadDim == 0,
            "glm5_next implements NoPE latent attention only")

        heads = deepseek.attentionHeads
        qHeadDim = deepseek.qkNopeHeadDim
        kvLoraRank = deepseek.kvLoraRank
        vHeadDim = deepseek.vHeadDim
        qkNopeHeadDim = deepseek.qkNopeHeadDim
        scale = pow(Float(qHeadDim), -0.5)

        _qAProj.wrappedValue = Linear(
            deepseek.hiddenSize, deepseek.qLoraRank, bias: deepseek.attentionBias)
        _qALayerNorm.wrappedValue = RMSNorm(dimensions: deepseek.qLoraRank, eps: 1e-6)
        _qBProj.wrappedValue = Linear(deepseek.qLoraRank, heads * qHeadDim, bias: false)
        _kvAProjWithMQA.wrappedValue = Linear(
            deepseek.hiddenSize, kvLoraRank, bias: deepseek.attentionBias)
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: 1e-6)
        _embedQ.wrappedValue = MultiLinear(
            inputDimensions: qkNopeHeadDim, outputDimensions: kvLoraRank, numHeads: heads)
        _unembedOut.wrappedValue = MultiLinear(
            inputDimensions: kvLoraRank, outputDimensions: vHeadDim, numHeads: heads)
        _oProj.wrappedValue = Linear(
            heads * vHeadDim, deepseek.hiddenSize, bias: deepseek.attentionBias)
        _indexer.wrappedValue = Glm5NextIndexer(config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: CacheList?) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let queryRank = qALayerNorm(qAProj(x))
        var q = qBProj(queryRank).reshaped(B, L, heads, qHeadDim).transposed(0, 2, 1, 3)

        var kvLatent = kvALayerNorm(kvAProjWithMQA(x)).expandedDimensions(axis: 1)
        let latentCache = cache?.children.first
        let indexerCache = cache?.children.last
        if let latentCache {
            (kvLatent, _) = latentCache.update(keys: kvLatent, values: kvLatent)
        }

        let topKIndices = indexer(
            x, queryRank: queryRank, mask: mask, cache: indexerCache)

        var attentionMask = mask
        if let topKIndices {
            let Kv = kvLatent.dim(2)
            let selectedValid = topKIndices .>= 0
            if L == 1 {
                let clamped = clip(topKIndices[0..., 0..., 0, 0...], min: 0, max: Kv - 1)
                let gatherIndices = clamped[.ellipsis, .newAxis]
                kvLatent = takeAlong(
                    kvLatent,
                    broadcast(
                        gatherIndices,
                        to: Array(gatherIndices.shape.dropLast()) + [kvLatent.dim(-1)]),
                    axis: 2)
                var selection = selectedValid[0..., 0..., 0, 0...][0..., 0..., .newAxis, 0...]
                if let mask, mask.dtype == .bool {
                    let keys = mask.reshaped(B, -1, Kv)[0..., 0, 0...]
                    let gathered = takeAlong(
                        broadcast(
                            keys[0..., .newAxis, 0...], to: [B, clamped.dim(1), Kv]),
                        clamped, axis: -1)
                    selection = selection .&& gathered[0..., 0..., .newAxis, 0...]
                }
                attentionMask = selection
            } else {
                var shape = topKIndices.shape
                shape[shape.count - 1] = Kv + 1
                let safeIndices = MLX.where(selectedValid, topKIndices, MLXArray(Int32(Kv)))
                var sparseMask = MLXArray.zeros(shape, dtype: .bool)
                sparseMask = putAlong(
                    sparseMask, safeIndices, values: MLXArray(true), axis: -1)
                sparseMask = sparseMask[.ellipsis, ..<Kv]
                if let mask, mask.dtype == .bool {
                    sparseMask = sparseMask .&& mask
                }
                attentionMask = sparseMask
            }
        }

        let absorbed =
            L == 1
            || L
                <= maxAbsorbedQueries(
                    kvLoraRank: kvLoraRank, qkNopeHeadDim: qkNopeHeadDim,
                    vHeadDim: vHeadDim, cacheLength: latentLength(kvLatent))

        let keys: MLXArray
        let values: MLXArray
        if absorbed {
            q = embedQ(q)
            keys = kvLatent
            values = kvLatent
        } else {
            keys = embedQ(kvLatent, transpose: false)
            values = unembedOut(kvLatent)
        }

        var output = MLXFast.scaledDotProductAttention(
            queries: q, keys: keys, values: values, scale: scale,
            mask: attentionMask.map { .array($0) } ?? .none)
        if absorbed {
            output = unembedOut(output)
        }

        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// MARK: - Stack

class Glm5NextDecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") var attention: Glm5NextAttentionLayer
    @ModuleInfo var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attn_hc") var attentionHyperConnection: HyperConnection
    @ModuleInfo(key: "ffn_hc") var ffnHyperConnection: HyperConnection

    init(_ config: Glm5NextConfiguration.TextConfiguration, layerIndex: Int) {
        isLinear = config.isLinear(layerIndex: layerIndex)
        _attention.wrappedValue =
            isLinear
            ? Glm5NextLinearAttention(config) : Glm5NextSparseAttention(config)
        _mlp.wrappedValue =
            config.isSparseMLP(layerIndex: layerIndex)
            ? DeepseekV32MoE(config.deepseek) : DeepseekV32MLP(config.deepseek)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.deepseek.hiddenSize, eps: config.deepseek.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.deepseek.hiddenSize, eps: config.deepseek.rmsNormEps)
        _attentionHyperConnection.wrappedValue = HyperConnection(
            hiddenSize: config.deepseek.hiddenSize,
            multiplier: config.hcMultiplier,
            rmsNormEps: config.deepseek.rmsNormEps,
            eps: config.hcEps,
            sinkhornIterations: config.hcSinkhornIterations)
        _ffnHyperConnection.wrappedValue = HyperConnection(
            hiddenSize: config.deepseek.hiddenSize,
            multiplier: config.hcMultiplier,
            rmsNormEps: config.deepseek.rmsNormEps,
            eps: config.hcEps,
            sinkhornIterations: config.hcSinkhornIterations)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        var residual = x
        var (collapsed, post, comb) = attentionHyperConnection(x)
        let attended = attention.attend(
            inputLayerNorm(collapsed), mask: mask, cache: cache)
        var h = hyperConnectionExpand(attended, residual: residual, post: post, comb: comb)

        residual = h
        (collapsed, post, comb) = ffnHyperConnection(h)
        let projected = mlp(postAttentionLayerNorm(collapsed))
        h = hyperConnectionExpand(projected, residual: residual, post: post, comb: comb)
        return h
    }
}

/// The two shapes a layer's `self_attn` slot can take. One property, one checkpoint key.
protocol Glm5NextAttentionLayer: Module {
    func attend(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray
}

extension Glm5NextLinearAttention: Glm5NextAttentionLayer {
    func attend(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        callAsFunction(x, mask: mask, cache: cache as? ArraysCache)
    }
}

extension Glm5NextSparseAttention: Glm5NextAttentionLayer {
    func attend(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        callAsFunction(x, mask: mask, cache: cache as? CacheList)
    }
}

class Glm5NextInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Glm5NextDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    let hcMultiplier: Int
    let recurrentIndex: Int
    let attentionIndex: Int

    init(_ config: Glm5NextConfiguration.TextConfiguration) {
        hcMultiplier = config.hcMultiplier
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.deepseek.vocabularySize,
            dimensions: config.deepseek.hiddenSize)
        _layers.wrappedValue = (0 ..< config.deepseek.hiddenLayers).map {
            Glm5NextDecoderLayer(config, layerIndex: $0)
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: config.deepseek.hiddenSize, eps: config.deepseek.rmsNormEps)
        recurrentIndex =
            (0 ..< config.deepseek.hiddenLayers).first { config.isLinear(layerIndex: $0) } ?? 0
        attentionIndex =
            (0 ..< config.deepseek.hiddenLayers).first { !config.isLinear(layerIndex: $0) } ?? 0
        super.init()
    }

    func callAsFunction(
        inputs: MLXArray?, inputEmbedding: MLXArray?, cache: [KVCache]?
    ) -> MLXArray {
        guard var h = inputEmbedding ?? inputs.map({ embedTokens($0) }) else {
            fatalError("either inputs or inputEmbedding must be provided")
        }

        let attentionCache = cache.flatMap {
            $0.indices.contains(attentionIndex) ? $0[attentionIndex] : nil
        }
        let attentionMask = createCausalMaskArray(
            h: h, cache: (attentionCache as? CacheList)?.children.first)
        let recurrentMask = cache.flatMap {
            $0.indices.contains(recurrentIndex)
                ? createSSMMask(h: h, cache: $0[recurrentIndex] as? MambaCache) : nil
        }

        // Every layer reads and writes `hcMultiplier` parallel residual streams.
        h = broadcast(
            h[0..., 0..., .newAxis, 0...],
            to: [h.dim(0), h.dim(1), hcMultiplier, h.dim(2)])

        for (i, layer) in layers.enumerated() {
            h = layer(
                h, mask: layer.isLinear ? recurrentMask : attentionMask, cache: cache?[i])
        }

        return norm(h.mean(axis: 2))
    }
}

/// The array form of the causal mask, which is what the sparse selection has to be
/// combined with.
private func createCausalMaskArray(h: MLXArray, cache: KVCache?) -> MLXArray? {
    let n = h.dim(1)
    guard n > 1 else { return nil }
    let offset = cache?.offset ?? 0
    return createCausalMask(n: n, offset: offset)
}

// MARK: - Language model

public class Glm5NextLanguageModel: Module, LLMModel {

    public let vocabularySize: Int

    @ModuleInfo var model: Glm5NextInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let config: Glm5NextConfiguration.TextConfiguration

    public init(_ config: Glm5NextConfiguration.TextConfiguration) {
        self.config = config
        vocabularySize = config.deepseek.vocabularySize
        _model.wrappedValue = Glm5NextInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                config.deepseek.hiddenSize, config.deepseek.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(inputs: inputs, inputEmbedding: nil, cache: cache)
    }

    public func logits(
        inputs: MLXArray?, inputEmbedding: MLXArray?, cache: [KVCache]?
    ) -> MLXArray {
        let out = model(inputs: inputs, inputEmbedding: inputEmbedding, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    /// Two slots for the recurrent layers, and a latent cache plus the indexer's own for
    /// the attending ones.
    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        model.layers.map { layer -> KVCache in
            layer.isLinear
                ? MambaCache()
                : CacheList(
                    MaterializedKVCache(KVCacheSimple()), MaterializedKVCache(KVCacheSimple()))
        }
    }

    /// Moves the checkpoint onto the module tree.
    ///
    /// The hyper-connection blocks are stored with their role in the name rather than as
    /// submodules, the linear attention's three convolutions are stored separately and the
    /// module wants them fused, and the forget gate's parameters sit directly on
    /// `self_attn`.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights.filter { !$0.key.contains("mtp.") }

        let layers = config.deepseek.hiddenLayers
        sanitized = sanitized.filter { key, _ in
            let parts = key.split(separator: ".")
            guard parts.count >= 3, parts[1] == "layers", let layer = Int(parts[2]) else {
                return true
            }
            return layer < layers
        }

        var dequantized: [String: MLXArray] = [:]
        for (key, value) in sanitized {
            guard key.contains("weight_scale_inv") else {
                if dequantized[key] == nil { dequantized[key] = value }
                continue
            }
            let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
            guard let weight = sanitized[weightKey] else {
                dequantized[key] = value
                continue
            }
            dequantized[weightKey] = DeepseekV32Model.dequantizeBlockScaled(
                weight, scaleInverse: value)
        }
        sanitized = dequantized

        DeepseekV32Model.stackExperts(weights: &sanitized, config: config.deepseek)
        DeepseekV32Model.splitFusedKV(weights: &sanitized, config: config.deepseek)

        let forgetGateParts = [
            "A_log", "dt_bias", "f_a_proj.weight", "f_b_proj.weight",
        ]
        var remapped: [String: MLXArray] = [:]
        var convolutionParts: [String: [String: MLXArray]] = [:]

        for (rawKey, value) in sanitized {
            var key =
                rawKey
                .replacingOccurrences(of: ".hc_attn_", with: ".attn_hc.")
                .replacingOccurrences(of: ".hc_ffn_", with: ".ffn_hc.")

            var fused = false
            for part in ["q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"] {
                let suffix = ".self_attn." + part
                if key.hasSuffix(suffix) {
                    let prefix = String(key.dropLast(part.count))
                    convolutionParts[prefix, default: [:]][String(part.prefix(1))] = value
                    fused = true
                    break
                }
            }
            if fused { continue }

            for part in forgetGateParts where key.hasSuffix(".self_attn." + part) {
                key = String(key.dropLast(part.count)) + "forget_gate." + part
                break
            }

            remapped[key] = value
        }

        for (prefix, parts) in convolutionParts {
            if let q = parts["q"], let k = parts["k"], let v = parts["v"] {
                remapped[prefix + "conv1d.weight"] = concatenated([q, k, v], axis: 0)
            } else {
                for (name, weight) in parts {
                    remapped[prefix + name + "_conv1d.weight"] = weight
                }
            }
        }

        for (key, value) in remapped
        where key.contains("conv1d.weight") && value.ndim == 3 && value.dim(-1) != 1 {
            remapped[key] = value.movedAxis(source: 2, destination: 1)
        }

        return remapped
    }
}

extension Glm5NextLanguageModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Vision

private func glm5NextRotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
}

private func glm5NextApplyVisionRoPE(
    _ q: MLXArray, _ k: MLXArray, cos: MLXArray, sin: MLXArray
) -> (MLXArray, MLXArray) {
    let (qDtype, kDtype) = (q.dtype, k.dtype)
    let q32 = q.asType(.float32)
    let k32 = k.asType(.float32)
    let cos = cos.expandedDimensions(axis: -2).asType(.float32)
    let sin = sin.expandedDimensions(axis: -2).asType(.float32)
    return (
        (q32 * cos + glm5NextRotateHalf(q32) * sin).asType(qDtype),
        (k32 * cos + glm5NextRotateHalf(k32) * sin).asType(kDtype)
    )
}

class Glm5NextVisionPatchEmbed: Module, UnaryLayer {
    let channels: Int
    let temporalPatchSize: Int
    let patchSize: Int
    let embedDimensions: Int

    @ModuleInfo var proj: Conv3d

    init(_ config: Glm5NextConfiguration.VisionConfiguration) {
        channels = config.channels
        temporalPatchSize = config.temporalPatchSize
        patchSize = config.patchSize
        embedDimensions = config.hiddenSize
        _proj.wrappedValue = Conv3d(
            inputChannels: config.channels, outputChannels: config.hiddenSize,
            kernelSize: .init((config.temporalPatchSize, config.patchSize, config.patchSize)),
            stride: .init((config.temporalPatchSize, config.patchSize, config.patchSize)),
            bias: true)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let reshaped =
            hiddenStates
            .reshaped(-1, channels, temporalPatchSize, patchSize, patchSize)
            .movedAxis(source: 1, destination: 4)
        return proj(reshaped).reshaped(-1, embedDimensions)
    }
}

/// The projector, whose clamped SwiGLU bounds the gate above and the up-projection on
/// both sides.
class Glm5NextVisionPatchMerger: Module, UnaryLayer {
    let limit: Float

    @ModuleInfo var proj: Linear
    @ModuleInfo(key: "post_projection_norm") var postProjectionNorm: LayerNorm
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dimensions: Int, contextDimensions: Int, limit: Float, bias: Bool = false) {
        self.limit = limit
        _proj.wrappedValue = Linear(dimensions, dimensions, bias: bias)
        _postProjectionNorm.wrappedValue = LayerNorm(dimensions: dimensions)
        _gateProj.wrappedValue = Linear(dimensions, contextDimensions, bias: bias)
        _upProj.wrappedValue = Linear(dimensions, contextDimensions, bias: bias)
        _downProj.wrappedValue = Linear(contextDimensions, dimensions, bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = gelu(postProjectionNorm(proj(x)))
        let gate = clip(gateProj(h), max: limit)
        let up = clip(upProj(h), min: -limit, max: limit)
        return downProj(silu(gate) * up)
    }
}

class Glm5NextVisionMLP: Module, UnaryLayer {
    let limit: Float

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: Glm5NextConfiguration.VisionConfiguration) {
        limit = config.swiGLULimit
        _gateProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.attentionBias)
        _upProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.attentionBias)
        _downProj.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: config.attentionBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = clip(gateProj(x), max: limit)
        let up = clip(upProj(x), min: -limit, max: limit)
        return downProj(silu(gate) * up)
    }
}

/// Attention over a flat sequence of patches, run one image at a time so patches never
/// attend across an image boundary.
class Glm5NextVisionAttention: Module {
    let heads: Int
    let scale: Float

    @ModuleInfo var qkv: Linear
    @ModuleInfo var proj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: Glm5NextConfiguration.VisionConfiguration) {
        heads = config.heads
        let headDim = config.hiddenSize / config.heads
        scale = pow(Float(headDim), -0.5)
        _qkv.wrappedValue = Linear(
            config.hiddenSize, config.hiddenSize * 3, bias: config.attentionBias)
        _proj.wrappedValue = Linear(
            config.hiddenSize, config.hiddenSize, bias: config.attentionBias)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray, boundaries: [Int],
        positionEmbeddings: (cos: MLXArray, sin: MLXArray)
    ) -> MLXArray {
        let length = hiddenStates.dim(0)
        let fused = qkv(hiddenStates)
            .reshaped(length, 3, heads, -1)
            .transposed(1, 0, 2, 3)

        var q = qNorm(fused[0])
        var k = kNorm(fused[1])
        let v = fused[2]

        (q, k) = glm5NextApplyVisionRoPE(
            q, k, cos: positionEmbeddings.cos, sin: positionEmbeddings.sin)

        let queries = q.transposed(1, 0, 2)[.newAxis, .ellipsis]
        let keys = k.transposed(1, 0, 2)[.newAxis, .ellipsis]
        let values = v.transposed(1, 0, 2)[.newAxis, .ellipsis]

        var outputs: [MLXArray] = []
        outputs.reserveCapacity(Swift.max(boundaries.count - 1, 1))
        for index in 0 ..< Swift.max(boundaries.count - 1, 0) {
            let (from, to) = (boundaries[index], boundaries[index + 1])
            guard to > from else { continue }
            outputs.append(
                MLXFast.scaledDotProductAttention(
                    queries: queries[0..., 0..., from ..< to, 0...],
                    keys: keys[0..., 0..., from ..< to, 0...],
                    values: values[0..., 0..., from ..< to, 0...],
                    scale: scale, mask: .none))
        }
        let attended =
            outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 2)
        return proj(attended.transposed(0, 2, 1, 3).reshaped(length, -1))
    }
}

class Glm5NextVisionBlock: Module {
    @ModuleInfo var norm1: RMSNorm
    @ModuleInfo var norm2: RMSNorm
    @ModuleInfo var attn: Glm5NextVisionAttention
    @ModuleInfo var mlp: Glm5NextVisionMLP

    init(_ config: Glm5NextConfiguration.VisionConfiguration) {
        _norm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _norm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attn.wrappedValue = Glm5NextVisionAttention(config)
        _mlp.wrappedValue = Glm5NextVisionMLP(config)
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray, boundaries: [Int],
        positionEmbeddings: (cos: MLXArray, sin: MLXArray)
    ) -> MLXArray {
        let h =
            hiddenStates
            + attn(
                norm1(hiddenStates), boundaries: boundaries,
                positionEmbeddings: positionEmbeddings)
        return h + mlp(norm2(h))
    }
}

class Glm5NextVisionModel: Module {
    let config: Glm5NextConfiguration.VisionConfiguration
    let rotaryDimensions: Int

    @ModuleInfo(key: "patch_embed") var patchEmbed: Glm5NextVisionPatchEmbed
    @ModuleInfo var blocks: [Glm5NextVisionBlock]
    @ModuleInfo var merger: Glm5NextVisionPatchMerger
    @ModuleInfo var downsample: Conv2d
    @ModuleInfo(key: "post_layernorm") var postLayerNorm: RMSNorm

    init(_ config: Glm5NextConfiguration.VisionConfiguration) {
        self.config = config
        rotaryDimensions = (config.hiddenSize / config.heads) / 2
        _patchEmbed.wrappedValue = Glm5NextVisionPatchEmbed(config)
        _blocks.wrappedValue = (0 ..< config.depth).map { _ in
            Glm5NextVisionBlock(config)
        }
        _merger.wrappedValue = Glm5NextVisionPatchMerger(
            dimensions: config.outHiddenSize,
            contextDimensions: config.projectionIntermediateSize,
            limit: config.swiGLULimit)
        _downsample.wrappedValue = Conv2d(
            inputChannels: config.hiddenSize, outputChannels: config.outHiddenSize,
            kernelSize: .init(config.spatialMergeSize),
            stride: .init(config.spatialMergeSize), bias: true)
        _postLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    /// Rotary frequencies for a square grid of the largest side present, indexed by each
    /// patch's row and column *within its merge block* -- which is why the ids are
    /// reshaped and transposed rather than simply enumerated.
    func rotaryPositionEmbeddings(grid: [[Int]]) -> (cos: MLXArray, sin: MLXArray) {
        let merge = config.spatialMergeSize
        var rows: [MLXArray] = []

        for entry in grid {
            let (t, h, w) = (entry[0], entry[1], entry[2])
            var hPositions = repeated(
                MLXArray(0 ..< h).expandedDimensions(axis: 1), count: w, axis: 1)
            hPositions =
                hPositions
                .reshaped(h / merge, merge, w / merge, merge)
                .transposed(0, 2, 1, 3)
                .flattened()

            var wPositions = repeated(
                MLXArray(0 ..< w).expandedDimensions(axis: 0), count: h, axis: 0)
            wPositions =
                wPositions
                .reshaped(h / merge, merge, w / merge, merge)
                .transposed(0, 2, 1, 3)
                .flattened()

            let stacked = MLX.stacked([hPositions, wPositions], axis: -1)
            rows.append(tiled(stacked, repetitions: [t, 1]))
        }

        let positionIDs = concatenated(rows, axis: 0)
        let maxSide = grid.flatMap { [$0[1], $0[2]] }.max() ?? 1

        let inverseFrequencies =
            1
            / pow(
                MLXArray(Float(10000)),
                MLXArray(stride(from: 0, to: rotaryDimensions, by: 2).map(Float.init))
                    / Float(rotaryDimensions))
        let frequencies = outer(
            MLXArray(0 ..< maxSide).asType(.float32), inverseFrequencies)

        let gathered = frequencies[positionIDs].reshaped(positionIDs.dim(0), -1)
        let embedding = concatenated([gathered, gathered], axis: -1)
        return (cos(embedding), sin(embedding))
    }

    func callAsFunction(_ hiddenStates: MLXArray, grid: [[Int]]) -> MLXArray {
        var h = patchEmbed(hiddenStates)
        let positionEmbeddings = rotaryPositionEmbeddings(grid: grid)

        // One boundary per frame: a video's frames each attend on their own.
        var boundaries = [0]
        for entry in grid {
            let frame = entry[1] * entry[2]
            for _ in 0 ..< entry[0] {
                boundaries.append(boundaries[boundaries.count - 1] + frame)
            }
        }

        for block in blocks {
            h = block(h, boundaries: boundaries, positionEmbeddings: positionEmbeddings)
        }
        h = postLayerNorm(h)

        let merge = config.spatialMergeSize
        h = h.reshaped(-1, merge, merge, h.dim(-1))
        h = downsample(h).reshaped(-1, config.outHiddenSize)
        return merger(h)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains("position_ids") { continue }
            if key.contains("patch_embed.proj.weight") || key.contains("downsample.weight"),
                !Self.isMLXConvolution(value)
            {
                if value.ndim == 5 {
                    sanitized[key] = value.transposed(0, 2, 3, 4, 1)
                } else if value.ndim == 4 {
                    sanitized[key] = value.transposed(0, 2, 3, 1)
                } else {
                    sanitized[key] = value
                }
            } else {
                sanitized[key] = value
            }
        }
        return sanitized
    }

    private static func isMLXConvolution(_ array: MLXArray) -> Bool {
        switch array.ndim {
        case 4:
            let (out, kH, kW) = (array.dim(0), array.dim(1), array.dim(2))
            return out >= kH && out >= kW && kH == kW
        case 5:
            let (out, kH, kW) = (array.dim(0), array.dim(2), array.dim(3))
            return out >= kH && out >= kW && kH == kW
        default:
            return false
        }
    }
}

// MARK: - Model

public class Glm5Next: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_model") private var visionModel: Glm5NextVisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Glm5NextLanguageModel

    public let config: Glm5NextConfiguration

    public var vocabularySize: Int { config.textConfiguration.deepseek.vocabularySize }
    public var kvHeads: [Int] {
        Array(
            repeating: config.textConfiguration.deepseek.attentionHeads,
            count: config.textConfiguration.deepseek.hiddenLayers)
    }

    public init(_ config: Glm5NextConfiguration) {
        self.config = config
        _visionModel.wrappedValue = Glm5NextVisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = Glm5NextLanguageModel(config.textConfiguration)
    }

    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        try languageModel.newCache(configuration: cacheConfiguration)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel.logits(inputs: inputs, inputEmbedding: nil, cache: cache)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?,
        prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }

        let embeddings = try input.image.map { image -> MLXArray in
            guard let frames = image.frames, !frames.isEmpty else {
                throw VLMError.imageRequired
            }
            let grid = frames.map { [$0.t, $0.h, $0.w] }
            let features = visionModel(
                image.pixels.asType(visionModel.patchEmbed.proj.weight.dtype), grid: grid)
            return merged(
                embedded: languageModel.model.embedTokens(tokens),
                features: features.asType(languageModel.model.embedTokens.weight.dtype),
                inputIDs: tokens)
        }

        let total = embeddings?.dim(1) ?? tokens.dim(1)

        func forward(_ range: Range<Int>) -> MLXArray {
            if let embeddings {
                return languageModel.logits(
                    inputs: nil, inputEmbedding: embeddings[0..., range], cache: cache)
            }
            return languageModel.logits(
                inputs: tokens[0..., range], inputEmbedding: nil, cache: cache)
        }

        let processed = try prefill.forEachChunk(total: total) { range in
            _ = forward(range)
            asyncEval(cache)
        }
        if processed > 0 {
            eval(cache)
        }

        prefill.progress?(total, total)
        return .logits(LMOutput(logits: forward(processed ..< total)))
    }

    /// Image positions take the patch features in order. Video tokens mark the positions
    /// when no image token does.
    func merged(embedded: MLXArray, features: MLXArray, inputIDs: MLXArray)
        -> MLXArray
    {
        var selected = inputIDs .== MLXArray(Int32(config.imageTokenID))
        eval(selected)
        if selected.asType(.int32).sum().item(Int.self) == 0 {
            selected = inputIDs .== MLXArray(Int32(config.videoTokenID))
        }

        let flat = embedded.reshaped(-1, embedded.dim(-1))
        let flatSelected = selected.reshaped(-1)
        let rank = clip(
            cumsum(flatSelected.asType(.int32), axis: 0) - 1,
            min: 0, max: features.dim(0) - 1)
        let source = takeAlong(
            features,
            broadcast(rank[0..., .newAxis], to: [flat.dim(0), features.dim(-1)]),
            axis: 0)
        return MLX.where(
            broadcast(flatSelected[0..., .newAxis], to: flat.shape), source, flat
        ).reshaped(embedded.shape)
    }

    /// Splits the checkpoint's flat container into the two halves and lets each sanitize
    /// its own keys.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var language: [String: MLXArray] = [:]
        var vision: [String: MLXArray] = [:]
        var other: [String: MLXArray] = [:]

        for (rawKey, value) in weights {
            var key = rawKey
            if key.hasPrefix("model.visual.") {
                key = "vision_model." + String(key.dropFirst("model.visual.".count))
            } else if key.hasPrefix("visual.") {
                key = "vision_model." + String(key.dropFirst("visual.".count))
            } else if key.hasPrefix("model.language_model.") {
                key =
                    "language_model.model."
                    + String(key.dropFirst("model.language_model.".count))
            } else if key.hasPrefix("lm_head.") {
                key = "language_model." + key
            }

            if key.hasPrefix("language_model.") {
                language[String(key.dropFirst("language_model.".count))] = value
            } else if key.hasPrefix("vision_model.") {
                vision[String(key.dropFirst("vision_model.".count))] = value
            } else {
                other[key] = value
            }
        }

        var out = other
        for (key, value) in languageModel.sanitize(weights: language) {
            out["language_model." + key] = value
        }
        for (key, value) in visionModel.sanitize(weights: vision) {
            out["vision_model." + key] = value
        }
        return out
    }
}

extension Glm5Next: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.loraLayers
    }
}
