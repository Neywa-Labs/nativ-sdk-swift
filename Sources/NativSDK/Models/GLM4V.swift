import CoreImage
import Foundation
import MLX
import MLXNN

private enum GLM4VSupport {
    final class VisionRotaryEmbedding {
        let inverseFreq: MLXArray

        init(dimensions: Int, theta: Float) {
            let p =
                MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
                / dimensions
            inverseFreq = 1.0 / pow(theta, p)
        }

        func callAsFunction(sequenceLength: Int) -> MLXArray {
            let seq = MLXArray(0 ..< sequenceLength).asType(inverseFreq.dtype)
            return outer(seq, inverseFreq)
        }
    }

}

// MARK: - Language

private enum Language {
    // MARK: M-RoPE helpers

    /// Apply rotary position embedding (language model style - interleaved).
    static func applyRotaryPosEmb(
        q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray,
    )
        -> (MLXArray, MLXArray)
    {
        // cos, sin: (batch, seq, headDim) -> (batch, 1, seq, headDim)
        var cos = cos[0..., .newAxis, 0..., 0...]
        var sin = sin[0..., .newAxis, 0..., 0...]

        // Take first half of frequencies, repeat-interleave to full headDim
        let halfDim = cos.dim(-1) / 2
        cos = repeated(cos[.ellipsis, ..<halfDim], count: 2, axis: -1)
        sin = repeated(sin[.ellipsis, ..<halfDim], count: 2, axis: -1)

        let rotaryDim = cos.dim(-1)
        let qRot = q[.ellipsis, ..<rotaryDim]
        let qPass = q[.ellipsis, rotaryDim...]
        let kRot = k[.ellipsis, ..<rotaryDim]
        let kPass = k[.ellipsis, rotaryDim...]

        let qEmbed = (qRot * cos) + (RotaryEmbeddingOps.rotateInterleaved(qRot) * sin)
        let kEmbed = (kRot * cos) + (RotaryEmbeddingOps.rotateInterleaved(kRot) * sin)

        return (
            concatenated([qEmbed, qPass], axis: -1),
            concatenated([kEmbed, kPass], axis: -1),
        )
    }

    // MARK: M-RoPE Rotary Embedding

    fileprivate final class GLM4VRotaryEmbedding {
        let mropeSplitIndices: [Int]
        let invFreq: MLXArray
        let attentionScaling: Float

        init(_ config: GLM4VConfiguration.TextConfiguration) {
            // Pre-compute cumulative split indices from mropeSection (config-derived, never changes)
            var indices = [Int]()
            var cumsum = 0
            for s in config.ropeParameters.mropeSection.dropLast() {
                cumsum += s
                indices.append(cumsum)
            }
            mropeSplitIndices = indices

            let dim = Int(Float(config.headDim) * config.ropeParameters.partialRotaryFactor)
            let base = config.ropeParameters.ropeTheta
            attentionScaling = 1.0

            let p =
                MLXArray(stride(from: 0, to: dim, by: 2)).asType(.int64).asType(.float32)
                / Float(dim)
            invFreq = 1.0 / pow(base, p)
        }

        /// Apply M-RoPE: select different frequency dimensions for T, H, W.
        func applyMrope(_ freqs: MLXArray) -> MLXArray {
            let chunks = split(freqs, indices: mropeSplitIndices, axis: -1)
            // Select chunk[i % 3] from the temporal dimension (axis 0)
            let selected = chunks.enumerated().map { i, chunk in
                chunk[i % 3]
            }
            return concatenated(selected, axis: -1)
        }

        func callAsFunction(_ x: MLXArray, positionIDs: MLXArray) -> (MLXArray, MLXArray) {
            // positionIds: (3, batch, seq)
            let batchSize = positionIDs.dim(1)

            // inv_freq: (dim/2,) -> (1, 1, dim/2, 1)
            var invFreqExpanded = invFreq[.newAxis, .newAxis, 0..., .newAxis].asType(.float32)
            // Broadcast to (3, batch, dim/2, 1)
            invFreqExpanded = broadcast(
                invFreqExpanded,
                to: [3, batchSize, invFreq.dim(0), 1],
            )

            // position_ids: (3, batch, seq) -> (3, batch, 1, seq)
            let positionIDsExpanded = positionIDs[0..., 0..., .newAxis, 0...].asType(.float32)

            // freqs = matmul -> (3, batch, dim/2, seq) -> transpose -> (3, batch, seq, dim/2)
            let freqs = matmul(invFreqExpanded, positionIDsExpanded).transposed(0, 1, 3, 2)

            // Apply M-RoPE section selection -> (batch, seq, dim/2)
            let mropeFreqs = applyMrope(freqs)

            // Double frequencies and compute cos/sin
            let emb = concatenated([mropeFreqs, mropeFreqs], axis: -1)
            let cos = MLX.cos(emb) * attentionScaling
            let sin = MLX.sin(emb) * attentionScaling

            return (cos.asType(x.dtype), sin.asType(x.dtype))
        }
    }

    // MARK: Attention

    fileprivate final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        init(_ args: GLM4VConfiguration.TextConfiguration) {
            let dim = args.hiddenSize
            heads = args.attentionHeads
            kvHeads = args.kvHeads
            headDim = args.headDim
            scale = pow(Float(headDim), -0.5)

            _wq.wrappedValue = Linear(dim, heads * headDim, bias: args.attentionBias)
            _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
            _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
            _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
            positionEmbeddings: (MLXArray, MLXArray),
        )
            -> MLXArray
        {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

            let (cos, sin) = positionEmbeddings
            (queries, keys) = applyRotaryPosEmb(q: queries, k: keys, cos: cos, sin: sin)

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    // MARK: MLP

    fileprivate final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_up_proj") var gateUpProj: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gateUpProj.wrappedValue = Linear(dimensions, hiddenDimensions * 2, bias: false)
            _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let x = gateUpProj(x)
            let parts = split(x, parts: 2, axis: -1)
            return down(silu(parts[0]) * parts[1])
        }
    }

    // MARK: Decoder Layer

    fileprivate final class GLM4VDecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_self_attn_layernorm") var postSelfAttnLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
        @ModuleInfo(key: "post_mlp_layernorm") var postMlpLayerNorm: RMSNorm

        init(_ args: GLM4VConfiguration.TextConfiguration) {
            _attention.wrappedValue = Attention(args)
            mlp = MLP(
                dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize,
            )
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps,
            )
            _postSelfAttnLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps,
            )
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps,
            )
            _postMlpLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps,
            )
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
            positionEmbeddings: (MLXArray, MLXArray),
        )
            -> MLXArray
        {
            var r = x
            var h = attention(
                inputLayerNorm(x), mask: mask, cache: cache,
                positionEmbeddings: positionEmbeddings,
            )
            h = postSelfAttnLayerNorm(h)
            h = r + h
            r = h
            h = postAttentionLayerNorm(h)
            h = mlp(h)
            h = postMlpLayerNorm(h)
            h = r + h
            return h
        }
    }

    // MARK: Text Model

    fileprivate final class GLM4VTextModel: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        fileprivate let layers: [GLM4VDecoderLayer]
        fileprivate let norm: RMSNorm
        let rotaryEmb: GLM4VRotaryEmbedding

        init(_ args: GLM4VConfiguration.TextConfiguration) {
            precondition(args.vocabularySize > 0)

            _embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize,
            )

            layers = (0 ..< args.hiddenLayers)
                .map { _ in GLM4VDecoderLayer(args) }
            norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
            rotaryEmb = GLM4VRotaryEmbedding(args)
        }

        func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil,
            positionIDs: MLXArray? = nil,
        )
            -> MLXArray
        {
            var h: MLXArray
            if let inputEmbedding {
                h = inputEmbedding
            } else if let inputs {
                h = embedTokens(inputs)
            } else {
                fatalError("one of inputs or inputEmbedding must be non-nil")
            }

            // Compute position_ids if not provided (autoregressive generation)
            var posIDs: MLXArray
            if let positionIDs {
                posIDs = positionIDs
            } else {
                let offset = cache?.first?.offset ?? 0
                let seqLen = h.dim(h.ndim - 2)
                let positions = MLXArray(Int32(offset) ..< Int32(offset + seqLen))
                    .expandedDimensions(axis: 0)
                posIDs = tiled(positions, repetitions: [3, 1, 1])
            }

            let positionEmbeddings = rotaryEmb(h, positionIDs: posIDs)
            let mask = createAttentionMask(h: h, cache: cache?.first)

            for (i, layer) in layers.enumerated() {
                h = layer(
                    h, mask: mask, cache: cache?[i],
                    positionEmbeddings: positionEmbeddings,
                )
            }

            return norm(h)
        }
    }

    // MARK: Language Model

    fileprivate final class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: GLM4VTextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int]
        var positionIDs: MLXArray?
        var ropeDeltas: MLXArray?

        init(_ args: GLM4VConfiguration.TextConfiguration) {
            model = GLM4VTextModel(args)

            if !args.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    args.hiddenSize, args.vocabularySize, bias: false,
                )
            }

            kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        }

        func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil,
        )
            -> LMOutput
        {
            var currentPositionIDs: MLXArray? = nil

            let cacheOffset: Int =
                if let cache, let first = cache.first {
                    first.offset
                } else {
                    0
                }

            if let storedPositionIDs = positionIDs {
                let seqLen: Int =
                    if let inputEmbedding {
                        inputEmbedding.dim(inputEmbedding.ndim - 2)
                    } else if let inputs {
                        inputs.dim(inputs.ndim - 1)
                    } else {
                        0
                    }

                let storedLen = storedPositionIDs.dim(2)
                if cacheOffset + seqLen <= storedLen {
                    // Prefill: use stored M-RoPE position IDs
                    currentPositionIDs =
                        storedPositionIDs[
                            0..., 0..., cacheOffset ..< (cacheOffset + seqLen),
                        ]
                } else {
                    // Autoregressive: compute sequential positions using rope_deltas
                    let delta = ropeDeltas ?? MLXArray(Int32(0))
                    let batchSize = inputEmbedding?.dim(0) ?? inputs?.dim(0) ?? 1
                    var posArrays = [MLXArray]()
                    for _ in 0 ..< 3 {
                        let pos = MLXArray(Int32(cacheOffset) ..< Int32(cacheOffset + seqLen))
                            .expandedDimensions(axis: 0)
                        let tiledPos = tiled(pos, repetitions: [batchSize, 1])
                        posArrays.append((tiledPos + delta).expandedDimensions(axis: 0))
                    }
                    currentPositionIDs = concatenated(posArrays, axis: 0)
                }
            }

            var out = model(
                inputs, cache: cache, inputEmbedding: inputEmbedding,
                positionIDs: currentPositionIDs,
            )
            if let lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }
            return LMOutput(logits: out)
        }
    }
}

// MARK: - Vision

enum GLM4Vision {
    fileprivate static func applyRotaryPosEmbVision(
        _ tensor: MLXArray, freqs: MLXArray,
    )
        -> MLXArray
    {
        var cosVal = MLX.cos(freqs)
        var sinVal = MLX.sin(freqs)

        // freqs: (seq, dim/2) -> expand to (seq, 1, dim) for broadcasting with (seq, heads, dim)
        cosVal = expandedDimensions(cosVal, axis: 1)
        cosVal = tiled(cosVal, repetitions: [1, 1, 2])

        sinVal = expandedDimensions(sinVal, axis: 1)
        sinVal = tiled(sinVal, repetitions: [1, 1, 2])

        let output = (tensor * cosVal) + (RotaryEmbeddingOps.rotateHalf(tensor) * sinVal)
        return output.asType(tensor.dtype)
    }

    fileprivate final class PatchEmbed: Module, UnaryLayer {
        @ModuleInfo var proj: Conv3d

        let patchSize: Int
        let temporalPatchSize: Int
        let inChannels: Int
        let embedDim: Int

        init(_ config: GLM4VConfiguration.VisionConfiguration) {
            patchSize = config.patchSize
            temporalPatchSize = config.temporalPatchSize
            inChannels = config.inChannels
            embedDim = config.hiddenSize

            let kernelSize = IntOrTriple(
                [config.temporalPatchSize, config.patchSize, config.patchSize],
            )
            _proj.wrappedValue = Conv3d(
                inputChannels: config.inChannels,
                outputChannels: config.hiddenSize,
                kernelSize: kernelSize,
                stride: kernelSize,
                bias: true,
            )
        }

        func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            var h = hiddenStates.reshaped(
                -1, inChannels, temporalPatchSize, patchSize, patchSize,
            ).movedAxis(source: 1, destination: 4)

            h = proj(h)
            h = h.reshaped(-1, embedDim)
            return h
        }
    }

    fileprivate final class Attention: Module {
        let numHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear
        init(_ config: GLM4VConfiguration.VisionConfiguration) {
            numHeads = config.numHeads
            headDim = config.hiddenSize / config.numHeads
            scale = pow(Float(headDim), -0.5)

            _qkv.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSize * 3, bias: false,
            )
            _proj.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSize, bias: false,
            )
        }

        func callAsFunction(
            _ x: MLXArray, cuSeqlens: [Int], rotaryPositionEmbedding: MLXArray,
        )
            -> MLXArray
        {
            let sequenceLength = x.dim(0)

            let qkvOut = qkv(x)
            let qkvReshaped = qkvOut.reshaped(sequenceLength, 3, numHeads, -1)
                .transposed(1, 0, 2, 3)
            let parts = split(qkvReshaped, parts: 3, axis: 0)
            var q = parts[0].squeezed(axis: 0)
            var k = parts[1].squeezed(axis: 0)
            let v = parts[2].squeezed(axis: 0)

            q = applyRotaryPosEmbVision(q, freqs: rotaryPositionEmbedding)
            k = applyRotaryPosEmbVision(k, freqs: rotaryPositionEmbedding)

            // Reshape for attention: (seq, heads, dim) -> (1, heads, seq, dim)
            let qT = q.transposed(1, 0, 2).expandedDimensions(axis: 0)
            let kT = k.transposed(1, 0, 2).expandedDimensions(axis: 0)
            let vT = v.transposed(1, 0, 2).expandedDimensions(axis: 0)

            // Per-image attention using cu_seqlens
            var attnOutputs = [MLXArray]()
            for i in 0 ..< (cuSeqlens.count - 1) {
                let start = cuSeqlens[i]
                let end = cuSeqlens[i + 1]
                let qChunk = qT[0..., 0..., start ..< end, 0...]
                let kChunk = kT[0..., 0..., start ..< end, 0...]
                let vChunk = vT[0..., 0..., start ..< end, 0...]
                let output = MLXFast.scaledDotProductAttention(
                    queries: qChunk, keys: kChunk, values: vChunk,
                    scale: scale, mask: .none,
                )
                attnOutputs.append(output)
            }

            let attnOutput = concatenated(attnOutputs, axis: 2)
                .transposed(0, 2, 1, 3)
                .reshaped(sequenceLength, -1)

            return proj(attnOutput)
        }
    }

    fileprivate final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        init(_ config: GLM4VConfiguration.VisionConfiguration) {
            _gate.wrappedValue = Linear(
                config.hiddenSize, config.outHiddenSize, bias: false,
            )
            _up.wrappedValue = Linear(
                config.hiddenSize, config.outHiddenSize, bias: false,
            )
            _down.wrappedValue = Linear(
                config.outHiddenSize, config.hiddenSize, bias: false,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate final class GLM4VVisionBlock: Module {
        @ModuleInfo var norm1: RMSNorm
        @ModuleInfo var norm2: RMSNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo var mlp: MLP

        init(_ config: GLM4VConfiguration.VisionConfiguration) {
            norm1 = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
            norm2 = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _attention.wrappedValue = Attention(config)
            mlp = MLP(config)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray, cuSeqlens: [Int], rotaryPositionEmbedding: MLXArray,
        )
            -> MLXArray
        {
            var hiddenStates =
                hiddenStates
                + attention(
                    norm1(hiddenStates),
                    cuSeqlens: cuSeqlens,
                    rotaryPositionEmbedding: rotaryPositionEmbedding,
                )
            hiddenStates = hiddenStates + mlp(norm2(hiddenStates))
            return hiddenStates
        }
    }

    fileprivate final class PatchMerger: Module, UnaryLayer {
        @ModuleInfo var proj: Linear
        @ModuleInfo(key: "post_projection_norm") var postProjectionNorm: LayerNorm
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        init(dim: Int, contextDim: Int) {
            _proj.wrappedValue = Linear(dim, dim, bias: false)
            _postProjectionNorm.wrappedValue = LayerNorm(dimensions: dim)
            _gate.wrappedValue = Linear(dim, contextDim, bias: false)
            _up.wrappedValue = Linear(dim, contextDim, bias: false)
            _down.wrappedValue = Linear(contextDim, dim, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = proj(x)
            h = gelu(postProjectionNorm(h))
            return down(silu(gate(h)) * up(h))
        }
    }

    fileprivate final class VisionEmbeddings: Module {
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        init(_ config: GLM4VConfiguration.VisionConfiguration) {
            let patchesPerSide = config.imageSize / config.patchSize
            _positionEmbedding.wrappedValue = Embedding(
                embeddingCount: patchesPerSide * patchesPerSide,
                dimensions: config.hiddenSize,
            )
        }

        func callAsFunction(
            _ hiddenStates: MLXArray, frames: [THW], imageTypeIDs: MLXArray,
        )
            -> MLXArray
        {
            let hiddenSize = positionEmbedding.weight.dim(1)
            let sourceSide = Int(sqrt(Double(positionEmbedding.weight.dim(0))))
            let source = positionEmbedding.weight
                .reshaped(sourceSide, sourceSide, hiddenSize)
                .transposed(2, 0, 1)
                .reshaped(1, hiddenSize, sourceSide, sourceSide)
                .asType(.float32)

            var parts = [MLXArray]()
            var offset = 0
            for frame in frames {
                let resized = bicubicInterpolate(source, size: (frame.h, frame.w))
                    .asType(positionEmbedding.weight.dtype)
                    .reshaped(hiddenSize, frame.h * frame.w)
                    .transposed(1, 0)

                let count = frame.product
                let coords = imageTypeIDs[offset ..< (offset + count), 0...].asType(.int32)
                let flatIndices = coords[0..., 0] * frame.w + coords[0..., 1]
                parts.append(resized[flatIndices])
                offset += count
            }

            let embeddings =
                parts.isEmpty
                ? MLXArray.zeros([0, hiddenSize], dtype: positionEmbedding.weight.dtype)
                : concatenated(parts, axis: 0)
            return hiddenStates + embeddings.asType(hiddenStates.dtype)
        }
    }

    final class VisionModel: Module {
        @ModuleInfo fileprivate var embeddings: VisionEmbeddings
        @ModuleInfo(key: "patch_embed") fileprivate var patchEmbed: PatchEmbed
        fileprivate let rotaryPosEmb: GLM4VSupport.VisionRotaryEmbedding
        @ModuleInfo(key: "blocks") fileprivate var blocks: [GLM4VVisionBlock]
        @ModuleInfo(key: "post_conv_layernorm") fileprivate var postConvLayernorm: RMSNorm
        @ModuleInfo fileprivate var downsample: Conv2d
        @ModuleInfo fileprivate var merger: PatchMerger
        @ModuleInfo(key: "post_layernorm") fileprivate var postLayernorm: RMSNorm

        let spatialMergeSize: Int

        init(_ config: GLM4VConfiguration.VisionConfiguration) {
            spatialMergeSize = config.spatialMergeSize

            _embeddings.wrappedValue = VisionEmbeddings(config)
            _patchEmbed.wrappedValue = PatchEmbed(config)

            let headDim = config.hiddenSize / config.numHeads
            rotaryPosEmb = GLM4VSupport.VisionRotaryEmbedding(
                dimensions: headDim / 2, theta: 10000,
            )

            _blocks.wrappedValue = (0 ..< config.depth).map { _ in
                GLM4VVisionBlock(config)
            }

            _postConvLayernorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps,
            )

            _downsample.wrappedValue = Conv2d(
                inputChannels: config.hiddenSize,
                outputChannels: config.outHiddenSize,
                kernelSize: IntOrPair(config.spatialMergeSize),
                stride: IntOrPair(config.spatialMergeSize),
                bias: true,
            )

            _merger.wrappedValue = PatchMerger(
                dim: config.outHiddenSize,
                contextDim: config.intermediateSize,
            )

            _postLayernorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps,
            )
        }

        func rotaryPositionEmbedding(_ frames: [THW]) -> (MLXArray, MLXArray) {
            var positionIDs = [MLXArray]()

            for row in frames {
                let (t, h, w) = row.values

                var hposIDs = expandedDimensions(MLXArray(0 ..< h), axis: 1)
                hposIDs = repeated(hposIDs, count: w, axis: 1)
                hposIDs =
                    hposIDs
                    .reshaped(
                        h / spatialMergeSize,
                        spatialMergeSize,
                        w / spatialMergeSize,
                        spatialMergeSize,
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                var wposIDs = expandedDimensions(MLXArray(0 ..< w), axis: 0)
                wposIDs = repeated(wposIDs, count: h, axis: 0)
                wposIDs =
                    wposIDs
                    .reshaped(
                        h / spatialMergeSize,
                        spatialMergeSize,
                        w / spatialMergeSize,
                        spatialMergeSize,
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                let stackedPosIDs = stacked([hposIDs, wposIDs], axis: -1)
                positionIDs.append(tiled(stackedPosIDs, repetitions: [t, 1]))
            }

            let indices = concatenated(positionIDs, axis: 0)
            let maxFrameSize = frames.lazy.map { max($0.h, $0.w) }.max() ?? 0
            let rotaryPosEmbFull = rotaryPosEmb(sequenceLength: maxFrameSize)[indices]

            return (rotaryPosEmbFull.reshaped(indices.dim(0), -1), indices)
        }

        func callAsFunction(_ hiddenStates: MLXArray, frames: [THW]) -> MLXArray {
            var hiddenStates = patchEmbed(hiddenStates)
            hiddenStates = postConvLayernorm(hiddenStates)
            let (rotaryPosEmbedding, imageTypeIDs) = rotaryPositionEmbedding(frames)
            hiddenStates = embeddings(hiddenStates, frames: frames, imageTypeIDs: imageTypeIDs)

            // Compute cu_seqlens from frames
            var cuSeqlens = [0]
            var cumsum = 0
            for frame in frames {
                for _ in 0 ..< frame.t {
                    cumsum += frame.h * frame.w
                    cuSeqlens.append(cumsum)
                }
            }

            for block in blocks {
                hiddenStates = block(
                    hiddenStates, cuSeqlens: cuSeqlens,
                    rotaryPositionEmbedding: rotaryPosEmbedding,
                )
            }

            hiddenStates = postLayernorm(hiddenStates)

            // Spatial merge via Conv2d downsample
            let hiddenDim = hiddenStates.dim(-1)
            hiddenStates = hiddenStates.reshaped(
                -1, spatialMergeSize, spatialMergeSize, hiddenDim,
            )
            hiddenStates = downsample(hiddenStates).reshaped(
                -1, downsample.weight.dim(0),
            )

            hiddenStates = merger(hiddenStates)
            return hiddenStates
        }

        func castPixelValues(_ pixels: MLXArray) -> MLXArray {
            pixels.asType(patchEmbed.proj.weight.dtype)
        }

        private func isMLXWeight(_ array: MLXArray) -> Bool {
            if array.ndim == 4 {
                let (outChannels, kH, kW) = (array.dim(0), array.dim(1), array.dim(2))
                return outChannels >= kH && outChannels >= kW && kH == kW
            } else if array.ndim == 5 {
                let (outChannels, kH, kW) = (array.dim(0), array.dim(2), array.dim(3))
                return outChannels >= kH && outChannels >= kW && kH == kW
            }
            return false
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()

            for (k, v) in weights {
                if k.contains("position_id") {
                    continue
                } else if k.contains("patch_embed.proj.weight")
                    || k.contains("downsample.weight")
                {
                    if isMLXWeight(v) {
                        sanitizedWeights[k] = v
                    } else {
                        if v.ndim == 5 {
                            sanitizedWeights[k] = v.transposed(0, 2, 3, 4, 1)
                        } else if v.ndim == 4 {
                            sanitizedWeights[k] = v.transposed(0, 2, 3, 1)
                        } else {
                            sanitizedWeights[k] = v
                        }
                    }
                } else {
                    sanitizedWeights[k] = v
                }
            }

            return sanitizedWeights
        }
    }
}

// MARK: - Processor

/// GLM4V VLM `UserInputProcessor`.
public struct GLM4VProcessor: UserInputProcessor {
    private let config: GLM4VProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: GLM4VProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    func preprocess(image: CIImage, resizedSize: CGSize) -> CIImage {
        image
            .toSRGB()
            .resampled(to: resizedSize, method: .bicubic)
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW,
    ) {
        let images = images.map { MediaProcessing.apply($0, processing: processing) }

        let size = images[0].extent.size
        let (resizedHeight, resizedWidth) = try QwenVL.targetSize(
            height: Int(size.height), width: Int(size.width),
            factor: config.patchSize * config.mergeSize,
            minPixels: config.minPixels, maxPixels: config.maxPixels,
        )
        let resizedSize = CGSize(width: resizedWidth, height: resizedHeight)

        let processedImages = images.map { image in
            preprocess(image: image, resizedSize: resizedSize).asMLXArray()
        }

        return try QwenVL.patchify(
            images: processedImages, mergeSize: config.mergeSize, patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize,
        )
    }

    /// Replace single image placeholder tokens with the correct count based on grid dimensions.
    private func replacePaddingTokens(
        in promptTokens: [Int], frames: [THW],
    )
        throws -> [Int]
    {
        let paddingToken = "<|image|>"
        let placeholderTokens = tokenizer.encode(
            text: "<|begin_of_image|>\(paddingToken)<|end_of_image|>",
        )
        let placeholderRanges = promptTokens.ranges(of: placeholderTokens)
        guard placeholderRanges.count == frames.count else {
            throw VLMError.processing(
                "Number of placeholder tokens (\(placeholderRanges.count)) does not match number of frames (\(frames.count))",
            )
        }
        let mergeLength = config.mergeSize * config.mergeSize
        let replacementSequences = frames.map { frame in
            let paddingCount = frame.product / mergeLength
            return tokenizer.encode(
                text:
                    "<|begin_of_image|>\(Array(repeating: paddingToken, count: paddingCount).joined())<|end_of_image|>",
            )
        }
        var result: [Int] = []
        var currentIndex = promptTokens.startIndex
        for (range, replacement) in zip(placeholderRanges, replacementSequences) {
            result.append(contentsOf: promptTokens[currentIndex ..< range.lowerBound])
            result.append(contentsOf: replacement)
            currentIndex = range.upperBound
        }
        if currentIndex < promptTokens.endIndex {
            result.append(contentsOf: promptTokens[currentIndex...])
        }
        return result
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = GLM4VMessageGenerator().generate(from: input)

        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext,
        )

        // Text-only input
        if input.images.isEmpty, input.videos.isEmpty {
            return LMInput(tokens: MLXArray(promptTokens))
        }

        // Process images
        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imagePixelsAndFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            let imagePixelsConcatenated = concatenated(imagePixelsAndFrames.map(\.0))
            processedImage = LMInput.ProcessedImage(
                pixels: imagePixelsConcatenated, frames: imagePixelsAndFrames.map(\.1),
            )
            if let imageFrames = processedImage?.frames {
                promptTokens = try replacePaddingTokens(
                    in: promptTokens, frames: imageFrames,
                )
            }
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage,
        )
    }
}

// MARK: - Model

/// GLM4V VLM
public final class GLM4V: Module, VLMModel, KVCacheDimensionProvider {
    public var toolCallFormat: ToolCallFormat? { .glm4 }

    @ModuleInfo(key: "vision_tower") private var visionModel: GLM4Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel

    public let config: GLM4VConfiguration

    public var vocabularySize: Int {
        config.textConfiguration.vocabularySize
    }

    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: GLM4VConfiguration) {
        self.config = config
        _visionModel.wrappedValue = GLM4Vision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = Language.LanguageModel(config.textConfiguration)
    }

    /// Compute 3D M-RoPE position IDs for prefill with images.
    /// Returns position_ids (3, batch, seq) and rope_deltas.
    private func getRopeIndex(
        inputIDs: MLXArray, imageGridThw: [THW]?,
    )
        -> (MLXArray, MLXArray)
    {
        let batchSize = inputIDs.dim(0)
        let seqLength = inputIDs.dim(1)
        let spatialMergeSize = config.visionConfiguration.spatialMergeSize
        let imageTokenID = config.baseConfiguration.imageTokenID

        guard let imageGridThw, !imageGridThw.isEmpty else {
            // Text only: sequential positions tiled 3x
            let positions = MLXArray(0 ..< Int32(seqLength)).expandedDimensions(axis: 0)
            let positionIDs = tiled(
                broadcast(positions, to: [batchSize, seqLength]).expandedDimensions(axis: 0),
                repetitions: [3, 1, 1],
            )
            let deltas = MLXArray(Int32(0))
            return (positionIDs, deltas)
        }

        // Build position_ids per batch element using 3 separate dimension arrays
        precondition(batchSize == 1, "GLM4V getRopeIndex only supports batchSize == 1")
        let positionIDs = zeros([3, batchSize, seqLength], type: Int32.self)
        var imageIndex = 0
        var mropePositionDelta = 0

        for batchIdx in 0 ..< batchSize {
            let inputTokens: [Int32] = inputIDs[batchIdx].asArray(Int32.self)

            // Keep 3 separate arrays for T, H, W dimensions
            var dimT = [Int32]()
            var dimH = [Int32]()
            var dimW = [Int32]()
            dimT.reserveCapacity(seqLength)
            dimH.reserveCapacity(seqLength)
            dimW.reserveCapacity(seqLength)
            var st = 0
            var lastMax: Int32 = -1

            // Append sequential text positions to all 3 dimensions
            let appendTextPositions = { (count: Int) in
                guard count > 0 else {
                    return
                }
                let base: Int32 = lastMax + 1
                for j in 0 ..< count {
                    let pos = base + Int32(j)
                    dimT.append(pos)
                    dimH.append(pos)
                    dimW.append(pos)
                }
                lastMax = base + Int32(count) - 1
            }

            // Process images: find each image token and build 3D positions
            while imageIndex < imageGridThw.count {
                guard let ed = inputTokens[st...].firstIndex(of: Int32(imageTokenID)) else {
                    break
                }

                let frame = imageGridThw[imageIndex]
                let llmGridT = frame.t
                let llmGridH = frame.h / spatialMergeSize
                let llmGridW = frame.w / spatialMergeSize
                imageIndex += 1

                // Text before this image
                appendTextPositions(ed - st)

                // Image tokens: 3D spatial positions
                let imgOffset: Int32 = lastMax + 1
                for t in 0 ..< llmGridT {
                    for h in 0 ..< llmGridH {
                        for w in 0 ..< llmGridW {
                            dimT.append(Int32(t) + imgOffset)
                            dimH.append(Int32(h) + imgOffset)
                            dimW.append(Int32(w) + imgOffset)
                        }
                    }
                }
                let tMax = Int32(llmGridT - 1) + imgOffset
                let hMax = Int32(llmGridH - 1) + imgOffset
                let wMax = Int32(llmGridW - 1) + imgOffset
                lastMax = max(tMax, max(hMax, wMax))

                st = ed + llmGridT * llmGridH * llmGridW
            }

            // Remaining text after last image
            appendTextPositions(inputTokens.count - st)

            positionIDs[0, batchIdx] = MLXArray(dimT)
            positionIDs[1, batchIdx] = MLXArray(dimH)
            positionIDs[2, batchIdx] = MLXArray(dimW)

            mropePositionDelta = Int(lastMax) + 1 - inputTokens.count
        }

        let deltas = MLXArray(Int32(mropePositionDelta))
        return (positionIDs, deltas)
    }

    private func inputEmbeddings(inputIDs: MLXArray, pixelValues: MLXArray?, frames: [THW]?)
        -> MLXArray
    {
        guard let pixelValues, let frames else {
            languageModel.positionIDs = nil
            languageModel.ropeDeltas = nil
            return languageModel.model.embedTokens(inputIDs[.newAxis, .ellipsis])
        }

        let inputEmbeds = languageModel.model.embedTokens(inputIDs)

        var hiddenStates = visionModel(pixelValues, frames: frames)

        if hiddenStates.ndim == 2 {
            hiddenStates = hiddenStates[.newAxis, 0..., 0...]
        }

        let merged = MultimodalEmbedding.mergeInputIDsWithImageFeatures(
            inputIDs: inputIDs, inputEmbeds: inputEmbeds, imageFeatures: hiddenStates,
            imageTokenID: config.baseConfiguration.imageTokenID,
            videoTokenID: config.baseConfiguration.videoTokenID,
        )

        let (positionIDs, ropeDeltas) = getRopeIndex(
            inputIDs: inputIDs, imageGridThw: frames,
        )
        languageModel.positionIDs = positionIDs
        languageModel.ropeDeltas = ropeDeltas

        return merged
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?,
        prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let dtype = visionModel.patchEmbed.proj.weight.dtype

        var allPixels: MLXArray?
        var allFrames: [THW] = []

        if let imagePixels = input.image?.pixels, let imageFrames = input.image?.frames {
            allPixels = imagePixels.asType(dtype)
            allFrames.append(contentsOf: imageFrames)
        }

        let inputEmbeddings = inputEmbeddings(
            inputIDs: input.text.tokens, pixelValues: allPixels,
            frames: allFrames.isEmpty ? nil : allFrames,
        )

        let result = languageModel(
            nil, cache: cache, inputEmbedding: inputEmbeddings,
        )

        prefill.progress?(inputEmbeddings.dim(1), inputEmbeddings.dim(1))
        return .logits(result)
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [any KVCache]?, state _: LMOutput.State?,
    )
        -> LMOutput
    {
        languageModel(input.tokens, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Step 1: Transform keys from HuggingFace format to internal format
        var transformed = [String: MLXArray]()
        for (key, value) in weights {
            var k = key

            // Map visual -> vision_tower
            if k.contains("visual"), !k.contains("vision_tower") {
                k = k.replacingOccurrences(of: "model.", with: "")
                k = k.replacingOccurrences(of: "visual", with: "vision_tower")
            }

            // Map model.language_model -> language_model.model
            if k.contains("model.language_model") {
                k = k.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model",
                )
            }

            // Map lm_head -> language_model.lm_head
            if k.contains("lm_head"), !k.hasPrefix("language_model") {
                k = k.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            }

            // The checkpoint includes a "next-n" prediction head at the layer index
            // equal to hiddenLayers (one past the last real decoder layer). Skip it.
            if k.contains("layers.\(config.textConfiguration.hiddenLayers).") {
                continue
            }

            transformed[k] = value
        }

        // Step 2: Sanitize vision weights (conv weight transposes)
        return visionModel.sanitize(weights: transformed)
    }
}

// MARK: - Configuration

/// Configuration for ``GLM4V``
public struct GLM4VConfiguration: Decodable, Sendable {
    public struct RopeParameters: Codable, Sendable {
        public let mropeSection: [Int]
        public let partialRotaryFactor: Float
        public let ropeTheta: Float

        enum CodingKeys: String, CodingKey {
            case mropeSection = "mrope_section"
            case partialRotaryFactor = "partial_rotary_factor"
            case ropeTheta = "rope_theta"
        }

        init(
            mropeSection: [Int] = [8, 12, 12],
            partialRotaryFactor: Float = 0.5,
            ropeTheta: Float = 10000,
        ) {
            self.mropeSection = mropeSection
            self.partialRotaryFactor = partialRotaryFactor
            self.ropeTheta = ropeTheta
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            mropeSection =
                try c.decodeIfPresent([Int].self, forKey: .mropeSection) ?? [8, 12, 12]
            partialRotaryFactor =
                try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.5
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        }
    }

    public struct TextConfiguration: Decodable, Sendable {
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let vocabularySize: Int
        public let ropeParameters: RopeParameters
        public let rmsNormEps: Float
        public var ropeTheta: Float {
            ropeParameters.ropeTheta
        }

        public let tieWordEmbeddings: Bool
        public let attentionBias: Bool

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case vocabularySize = "vocab_size"
            case ropeParameters = "rope_parameters"
            case ropeScaling = "rope_scaling"
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
            case rmsNormEps = "rms_norm_eps"
            case tieWordEmbeddings = "tie_word_embeddings"
            case attentionBias = "attention_bias"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 40
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 13696
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
            kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 2
            headDim =
                try c.decodeIfPresent(Int.self, forKey: .headDim)
                ?? (hiddenSize / attentionHeads)
            vocabularySize =
                try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_552
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
            tieWordEmbeddings =
                try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
            attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? true

            let decodedRope =
                try c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
                ?? c.decodeIfPresent(RopeParameters.self, forKey: .ropeScaling)
                ?? RopeParameters()
            let partial =
                try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
                ?? decodedRope.partialRotaryFactor
            let theta =
                try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? decodedRope.ropeTheta
            ropeParameters = RopeParameters(
                mropeSection: decodedRope.mropeSection,
                partialRotaryFactor: partial,
                ropeTheta: theta,
            )
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let depth: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHeads: Int
        public let patchSize: Int
        private let _imageSize: Int?
        public let outHiddenSize: Int
        public let spatialMergeSize: Int
        public let temporalPatchSize: Int
        private let _inChannels: Int?
        public var imageSize: Int {
            _imageSize ?? 336
        }

        public var inChannels: Int {
            _inChannels ?? 3
        }

        private let _rmsNormEps: Float?
        public var rmsNormEps: Float {
            _rmsNormEps ?? 1e-5
        }

        enum CodingKeys: String, CodingKey {
            case depth
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHeads = "num_heads"
            case patchSize = "patch_size"
            case _imageSize = "image_size"
            case outHiddenSize = "out_hidden_size"
            case spatialMergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
            case _inChannels = "in_channels"
            case _rmsNormEps = "rms_norm_eps"
        }
    }

    public struct BaseConfiguration: Codable, Sendable {
        public let modelType: String
        private let _vocabularySize: Int?
        private let _imageTokenID: Int?
        private let _videoTokenID: Int?
        private let _imageStartTokenID: Int?
        private let _hiddenSize: Int?

        public var vocabularySize: Int {
            _vocabularySize ?? 151_552
        }

        public var imageTokenID: Int {
            _imageTokenID ?? 151_343
        }

        public var videoTokenID: Int {
            _videoTokenID ?? 151_344
        }

        public var imageStartTokenID: Int {
            _imageStartTokenID ?? 151_339
        }

        public var hiddenSize: Int {
            _hiddenSize ?? 4096
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case _vocabularySize = "vocab_size"
            case _imageTokenID = "image_token_id"
            case _videoTokenID = "video_token_id"
            case _imageStartTokenID = "image_start_token_id"
            case _hiddenSize = "hidden_size"
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let baseConfiguration: BaseConfiguration

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        textConfiguration = try container.decode(
            TextConfiguration.self, forKey: .textConfiguration,
        )
        visionConfiguration = try container.decode(
            VisionConfiguration.self, forKey: .visionConfiguration,
        )

        // BaseConfiguration overlaid at top level (fields may be absent)
        baseConfiguration = try BaseConfiguration(from: decoder)
    }
}

/// Configuration for ``GLM4VProcessor``
public struct GLM4VProcessorConfiguration: Codable, Sendable {
    public struct Size: Codable, Sendable {
        public let shortestEdge: Int
        public let longestEdge: Int

        enum CodingKeys: String, CodingKey {
            case shortestEdge = "shortest_edge"
            case longestEdge = "longest_edge"
        }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let mergeSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let size: Size

    public var minPixels: Int {
        size.shortestEdge
    }

    public var maxPixels: Int {
        size.longestEdge
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case mergeSize = "merge_size"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case size
    }
}

// MARK: - Message Generator

/// Message Generator for GLM4V
public struct GLM4VMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> Message {
        [
            "role": message.role.rawValue,
            "content":
                message.images.map { _ in ["type": "image"] }
                + message.videos.map { _ in ["type": "video"] }
                + [["type": "text", "text": message.content]],
        ]
    }
}
