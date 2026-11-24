import Foundation
import MLX
import MLXNN

/// RMS normalization whose weight is stored as an offset from a per-site constant.
///
/// The weight is zero-initialized and the offset differs by position in the block, so a
/// single shared value silently changes every residual's scale.
class Plamo2RMSNorm: Module, UnaryLayer {

    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let offset: Float

    init(dimensions: Int, eps: Float, offset: Float = 1.0) {
        self.eps = eps
        self.offset = offset
        _weight.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight + offset, eps: eps)
    }
}

/// The two shapes a `mixer` slot can take.
protocol Plamo2Mixer: Module {
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray
}

class Plamo2Mamba: Module, Plamo2Mixer {

    let heads: Int
    let headDim: Int
    let stateSize: Int
    let convKernel: Int
    let intermediateSize: Int
    let dtDim: Int
    let eps: Float

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "conv1d") var conv: Conv1d
    @ModuleInfo(key: "bcdt_proj") var bcdtProj: Linear
    @ModuleInfo(key: "dt_proj") var dtProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "D") var D: MLXArray
    @ParameterInfo(key: "dt_norm_weight") var dtNormWeight: MLXArray
    @ParameterInfo(key: "B_norm_weight") var bNormWeight: MLXArray
    @ParameterInfo(key: "C_norm_weight") var cNormWeight: MLXArray

    init(_ args: Plamo2Configuration) {
        self.heads = args.mambaHeads
        self.headDim = args.hiddenSizePerHead
        self.stateSize = args.mambaStateSize
        self.convKernel = args.mambaConvKernel
        self.intermediateSize = args.mambaHeads * args.hiddenSizePerHead
        self.dtDim = Swift.max(64, args.hiddenSize / 16)
        self.eps = args.rmsNormEps

        _inProj.wrappedValue = Linear(
            args.hiddenSize, 2 * intermediateSize, bias: false)
        _conv.wrappedValue = Conv1d(
            inputChannels: intermediateSize, outputChannels: intermediateSize,
            kernelSize: convKernel, padding: 0, groups: intermediateSize, bias: false)
        _bcdtProj.wrappedValue = Linear(
            intermediateSize, dtDim + 2 * stateSize, bias: false)
        _dtProj.wrappedValue = Linear(dtDim, heads, bias: false)
        _outProj.wrappedValue = Linear(intermediateSize, args.hiddenSize, bias: false)

        _dtBias.wrappedValue = MLXArray.zeros([heads])
        _aLog.wrappedValue = log(MLXArray(1 ... heads).asType(.float32))
        _D.wrappedValue = MLXArray.ones([heads])
        _dtNormWeight.wrappedValue = MLXArray.ones([dtDim])
        _bNormWeight.wrappedValue = MLXArray.ones([stateSize])
        _cNormWeight.wrappedValue = MLXArray.ones([stateSize])
        super.init()
    }

    private func convolved(_ x: MLXArray, cache: MambaCache?, mask: MLXArray?)
        -> MLXArray
    {
        var input = x
        if let mask {
            input = MLX.where(mask[.ellipsis, .newAxis], input, MLXArray(0))
        }

        let padded: MLXArray
        if let cache {
            let tail =
                cache[0]
                ?? MLXArray.zeros(
                    [input.dim(0), convKernel - 1, intermediateSize], dtype: input.dtype)
            padded = concatenated([tail, input], axis: 1)
            let keep = convKernel - 1
            if let lengths = cache.currentLengths {
                let ends = clip(lengths, min: 0, max: padded.dim(1) - keep)
                let positions = (ends[0..., .newAxis, 0...] + MLXArray(0 ..< keep))
                    .reshaped(input.dim(0), keep, 1)
                cache[0] = takeAlong(
                    padded,
                    broadcast(
                        positions, to: [input.dim(0), keep, intermediateSize]
                    ).asType(.int32), axis: 1)
            } else {
                cache[0] = padded[0..., (1 - convKernel)...]
            }
        } else {
            padded = MLX.padded(
                input,
                widths: [
                    IntOrPair((0, 0)), IntOrPair((convKernel - 1, 0)), IntOrPair((0, 0)),
                ])
        }

        return silu(conv(padded))
    }

    func callAsFunction(
        _ x: MLXArray, mask _: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?, cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let slots = cache as? MambaCache

        // Per head, not two contiguous halves: the projection is reshaped to
        // [batch, length, heads, 2 * headDim] before the split.
        let mixed = inProj(x).reshaped(batch, length, heads, -1)
        let parts = split(mixed, indices: [headDim], axis: -1)
        let gate = parts[0]
        var hidden = parts[1].reshaped(batch, length, heads * headDim)

        hidden = convolved(hidden, cache: slots, mask: ssmMask)

        let projected = bcdtProj(hidden)
        let bcdt = split(projected, indices: [stateSize, 2 * stateSize], axis: -1)
        let B = MLXFast.rmsNorm(bcdt[0], weight: bNormWeight, eps: eps)
        let C = MLXFast.rmsNorm(bcdt[1], weight: cNormWeight, eps: eps)
        var dt = MLXFast.rmsNorm(bcdt[2], weight: dtNormWeight, eps: eps)
        dt = dtProj(dt)

        let (y, state) = ssmUpdate(
            hiddenStates: hidden.reshaped(batch, length, heads, headDim),
            ALog: aLog,
            B: B.reshaped(batch, length, 1, stateSize),
            C: C.reshaped(batch, length, 1, stateSize),
            D: D,
            dt: dt,
            dtBias: dtBias,
            state: slots?[1],
            mask: ssmMask,
            lengths: slots?.currentLengths)
        slots?[1] = state
        slots?.advance(length)

        let out = compiledSiluProduct(gate.flattened(start: 2), y.reshaped(batch, length, -1))
        return outProj(out)
    }
}

class Plamo2Attention: Module, Plamo2Mixer {

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ParameterInfo(key: "q_weight") var qWeight: MLXArray
    @ParameterInfo(key: "k_weight") var kWeight: MLXArray

    let rope: RoPE

    init(_ args: Plamo2Configuration, layerIndex: Int) {
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDim = args.hiddenSizePerHead
        self.scale = pow(Float(headDim), -0.5)

        _qkvProj.wrappedValue = Linear(
            args.hiddenSize, (heads + 2 * kvHeads) * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, args.hiddenSize, bias: false)
        _qWeight.wrappedValue = MLXArray.ones([heads, headDim])
        _kWeight.wrappedValue = MLXArray.ones([kvHeads, headDim])

        // Full-attention layers take `rope_theta`; every other layer takes the local base.
        let base =
            args.fullAttentionIndices.contains(layerIndex)
            ? args.ropeTheta : args.ropeLocalTheta
        self.rope = RoPE(dimensions: headDim, traditional: false, base: base)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, ssmMask _: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let (B, T) = (x.dim(0), x.dim(1))

        let queryWidth = heads * headDim
        let keyWidth = kvHeads * headDim
        let parts = split(
            qkvProj(x), indices: [queryWidth, queryWidth + keyWidth], axis: -1)

        var queries = parts[0].reshaped(B, T, heads, headDim).transposed(0, 2, 1, 3)
        var keys = parts[1].reshaped(B, T, kvHeads, headDim).transposed(0, 2, 1, 3)
        let values = parts[2].reshaped(B, T, kvHeads, headDim).transposed(0, 2, 1, 3)

        // Normalized without a weight, then scaled per head.
        queries =
            MLXFast.rmsNorm(queries, weight: MLXArray.mlxNone, eps: 1e-6)
            * qWeight[0..., .newAxis, 0...]
        keys =
            MLXFast.rmsNorm(keys, weight: MLXArray.mlxNone, eps: 1e-6)
            * kWeight[0..., .newAxis, 0...]

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache,
            scale: scale, mask: mask)

        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, T, -1))
    }
}

class Plamo2MLP: Module, UnaryLayer {

    @ModuleInfo(key: "gate_up_proj") var gateUpProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ args: Plamo2Configuration) {
        _gateUpProj.wrappedValue = Linear(
            args.hiddenSize, args.intermediateSize * 2, bias: false)
        _downProj.wrappedValue = Linear(
            args.intermediateSize, args.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let halves = split(gateUpProj(x), parts: 2, axis: -1)
        return downProj(compiledSiluProduct(halves[0], halves[1]))
    }
}

/// Four norms, each with its own offset, and the post-norms applied to the branch output
/// rather than the residual sum.
class Plamo2Block: Module {

    @ModuleInfo(key: "mixer") var mixer: Plamo2Mixer
    @ModuleInfo(key: "mlp") var mlp: Plamo2MLP
    @ModuleInfo(key: "pre_mixer_norm") var preMixerNorm: Plamo2RMSNorm
    @ModuleInfo(key: "post_mixer_norm") var postMixerNorm: Plamo2RMSNorm
    @ModuleInfo(key: "pre_mlp_norm") var preMLPNorm: Plamo2RMSNorm
    @ModuleInfo(key: "post_mlp_norm") var postMLPNorm: Plamo2RMSNorm

    let isMamba: Bool

    init(_ args: Plamo2Configuration, layerIndex: Int) {
        self.isMamba = args.isMamba(at: layerIndex)
        _mixer.wrappedValue =
            isMamba ? Plamo2Mamba(args) : Plamo2Attention(args, layerIndex: layerIndex)
        _mlp.wrappedValue = Plamo2MLP(args)
        _preMixerNorm.wrappedValue = Plamo2RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps, offset: 1)
        _postMixerNorm.wrappedValue = Plamo2RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps, offset: 1.0 / 5)
        _preMLPNorm.wrappedValue = Plamo2RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps, offset: 1)
        _postMLPNorm.wrappedValue = Plamo2RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps,
            offset: 1.0 / pow(5.0, 1.5))
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let mixed = mixer(preMixerNorm(x), mask: mask, ssmMask: ssmMask, cache: cache)
        let h = x + postMixerNorm(mixed)
        return h + postMLPNorm(mlp(preMLPNorm(h)))
    }
}

public class Plamo2Decoder: Module {

    let layers: [Plamo2Block]

    let recurrentIndex: Int?
    let attentionIndex: Int

    public init(_ args: Plamo2Configuration) {
        self.layers = (0 ..< args.hiddenLayers).map {
            Plamo2Block(args, layerIndex: $0)
        }
        self.recurrentIndex = args.mambaEnabled ? 0 : nil
        self.attentionIndex = args.mambaStep / 2
    }

    public func callAsFunction(_ x: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = x

        let attentionCache = cache.flatMap {
            $0.indices.contains(attentionIndex) ? $0[attentionIndex] : nil
        }
        let attentionMask = createAttentionMask(h: h, cache: attentionCache)
        let ssmMask = recurrentIndex.flatMap {
            createSSMMask(h: h, cache: cache?[$0] as? MambaCache)
        }

        for (i, layer) in layers.enumerated() {
            h = layer(
                h, mask: attentionMask, ssmMask: layer.isMamba ? ssmMask : nil,
                cache: cache?[i])
        }
        return h
    }
}

public class Plamo2ModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var decoder: Plamo2Decoder
    @ModuleInfo(key: "norm") var norm: Plamo2RMSNorm

    public init(_ args: Plamo2Configuration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        _decoder.wrappedValue = Plamo2Decoder(args)
        _norm.wrappedValue = Plamo2RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        callAsFunction(inputs: inputs, inputEmbedding: nil, cache: cache)
    }

    public func callAsFunction(
        inputs: MLXArray?, inputEmbedding: MLXArray?, cache: [KVCache]? = nil
    ) -> MLXArray {
        guard let h = inputEmbedding ?? inputs.map({ embedTokens($0) }) else {
            fatalError("either inputs or inputEmbedding must be provided")
        }
        return norm(decoder(h, cache: cache))
    }
}

public class Plamo2Model: Module, LLMModel {

    public let vocabularySize: Int

    public let model: Plamo2ModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Plamo2Configuration) {
        self.vocabularySize = args.vocabularySize
        self.model = Plamo2ModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(inputs: inputs, inputEmbedding: nil, cache: cache)
    }

    /// The forward a wrapping vision-language model needs, where the prompt arrives as
    /// merged embeddings rather than token ids.
    public func logits(
        inputs: MLXArray?, inputEmbedding: MLXArray?, cache: [KVCache]?
    ) -> MLXArray {
        let out = model(inputs: inputs, inputEmbedding: inputEmbedding, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        try model.decoder.layers.map {
            $0.isMamba ? MambaCache() : try makeAttentionKVCache(configuration: cacheConfiguration)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights
        for (key, value) in weights
        where key.contains("conv1d.weight") && value.dim(-1) != 1 {
            sanitized[key] = value.movedAxis(source: 2, destination: 1)
        }
        return sanitized
    }
}

public struct Plamo2Configuration: Decodable, Sendable {

    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var rmsNormEps: Float
    var tieWordEmbeddings: Bool
    var attentionHeads: Int
    var kvHeads: Int
    var hiddenSizePerHead: Int
    var maxPositionEmbeddings: Int
    var attentionWindowSize: Int
    var fullAttentionIndices: [Int]
    var ropeTheta: Float
    var ropeLocalTheta: Float
    var mambaStateSize: Int
    var mambaConvKernel: Int
    var mambaHeads: Int
    var mambaStep: Int
    var mambaChunkSize: Int
    var mambaEnabled: Bool
    var intermediateSize: Int
    var vocabularySize: Int

    /// Recurrent unless the layer index lands on the midpoint of the step -- with a special
    /// case for a stack shorter than that midpoint, where only the last layer attends.
    func isMamba(at layer: Int) -> Bool {
        guard mambaEnabled else { return false }
        precondition(mambaStep > 1, "mamba_step must exceed one")
        if hiddenLayers <= mambaStep / 2 {
            return layer != hiddenLayers - 1
        }
        return (layer % mambaStep) != (mambaStep / 2)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case rmsNormEps = "rms_norm_eps"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case hiddenSizePerHead = "hidden_size_per_head"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionWindowSize = "attention_window_size"
        case fullAttentionIndices = "full_attention_idx"
        case ropeTheta = "rope_theta"
        case ropeLocalTheta = "rope_local_theta"
        case mambaStateSize = "mamba_d_state"
        case mambaConvKernel = "mamba_d_conv"
        case mambaHeads = "mamba_num_heads"
        case mambaStep = "mamba_step"
        case mambaChunkSize = "mamba_chunk_size"
        case mambaEnabled = "mamba_enabled"
        case intermediateSize = "intermediate_size"
        case vocabularySize = "vocab_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "plamo2"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        hiddenSizePerHead =
            try c.decodeIfPresent(Int.self, forKey: .hiddenSizePerHead) ?? 128
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 2048
        attentionWindowSize =
            try c.decodeIfPresent(Int.self, forKey: .attentionWindowSize) ?? 2048
        fullAttentionIndices =
            try c.decodeIfPresent([Int].self, forKey: .fullAttentionIndices) ?? []
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        ropeLocalTheta =
            try c.decodeIfPresent(Float.self, forKey: .ropeLocalTheta) ?? 10_000
        mambaStateSize = try c.decodeIfPresent(Int.self, forKey: .mambaStateSize) ?? 64
        mambaConvKernel = try c.decodeIfPresent(Int.self, forKey: .mambaConvKernel) ?? 4
        mambaHeads = try c.decodeIfPresent(Int.self, forKey: .mambaHeads) ?? 64
        mambaStep = try c.decodeIfPresent(Int.self, forKey: .mambaStep) ?? 2
        mambaChunkSize = try c.decodeIfPresent(Int.self, forKey: .mambaChunkSize) ?? 256
        mambaEnabled = try c.decodeIfPresent(Bool.self, forKey: .mambaEnabled) ?? true
        intermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 13312
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 32000
    }
}

extension Plamo2Model: LoRAModel {
    public var loraLayers: [Module] {
        model.decoder.layers
    }
}
