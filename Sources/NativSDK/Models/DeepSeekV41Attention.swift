import Foundation
import MLX
import MLXNN

extension DeepSeekV41Language {
    /// Keeps the `index_topk` best compressed positions per query.
    ///
    /// A small side attention: query heads against one shared key per compressed
    /// position, scores rectified then combined by `weights_proj`. With a candidate
    /// source this is the second of two levels; the block selection is the first. Only
    /// compression-source layers publish index keys; every other indexer reads them from
    /// the shared handoff. Queries and keys pass through the QAT fake-quant before
    /// scoring.
    final class Indexer: Module {
        let ownsKeys: Bool
        let compressRatio: Int
        let isCandidateSource: Bool
        let usesCandidates: Bool
        let heads: Int
        let indexHeadDim: Int
        let ropeHeadDim: Int
        let indexTopK: Int
        let candidateTopKBlocks: Int
        let candidateBlockSize: Int
        let scale: Float
        let layerIndex: Int
        let rope: DeepSeekV41IndexRope

        @ModuleInfo(key: "wq_b") var wqB: Linear
        @ModuleInfo(key: "weights_proj") var weightsProj: Linear
        @ModuleInfo(key: "wk") var wk: Linear?
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

        init(_ config: DeepSeekV41Configuration, layerIndex: Int) {
            self.ownsKeys = config.kvSourceLayerIds.contains(layerIndex)
            self.compressRatio = config.compressRatios[layerIndex]
            self.isCandidateSource = layerIndex == config.candidateSourceLayerId
            self.usesCandidates =
                config.candidateSourceLayerId >= 0
                && config.candidateSourceLayerId < layerIndex
            self.heads = config.indexHeads
            self.indexHeadDim = config.indexHeadDim
            self.ropeHeadDim = config.qkRopeHeadDim
            self.indexTopK = config.indexTopK
            self.candidateTopKBlocks = config.candidateTopKBlocks
            self.candidateBlockSize = config.candidateBlockSize
            self.scale = pow(Float(config.indexHeadDim), -0.5)
            self.layerIndex = layerIndex
            self.rope = DeepSeekV41IndexRope(
                ropeDimensions: config.qkRopeHeadDim, theta: config.compressRopeTheta,
                yarn: DeepSeekV41Language.yarnParameters(config))

            _wqB.wrappedValue = Linear(
                config.qLoraRank, config.indexHeads * config.indexHeadDim, bias: false)
            _weightsProj.wrappedValue = Linear(
                config.hiddenSize, config.indexHeads, bias: false)
            if ownsKeys {
                _wk.wrappedValue = Linear(
                    config.headDim, config.indexHeadDim, bias: false)
                _kNorm.wrappedValue = RMSNorm(
                    dimensions: config.indexHeadDim, eps: config.rmsNormEps)
            }
            super.init()
        }

        /// Turn this layer's unrotated latent into index keys and publish them.
        @discardableResult
        func publishKeys(
            latent: MLXArray, startPosition: Int, cache: DeepSeekV41Cache
        ) -> MLXArray {
            guard let wk, let kNorm else {
                preconditionFailure("only an index-key owner can publish keys")
            }
            let count = latent.dim(1)
            let base = startPosition / compressRatio
            let positions = (MLXArray(0 ..< count) + base) * compressRatio
            let (cos, sin) = rope.table(
                length: max(
                    (base + count - 1) * compressRatio + 1, startPosition + count + 1))
            var keys = kNorm(wk(latent))
            keys = DeepSeekV41IndexScores.applyRotary(
                keys,
                cos: take(cos, positions, axis: 0)[.newAxis],
                sin: take(sin, positions, axis: 0)[.newAxis],
                ropeDimensions: ropeHeadDim)
            keys = DeepSeekV41FakeQuant.fp4UE8M0(keys)
            return cache.appendIndexKeys(layer: layerIndex, values: keys)
        }

        /// `latent` is this layer's rope-free compressed latent, nil when the layer does
        /// not compress or when its current group is still incomplete. An index-key owner
        /// turns it into index keys here, which has to happen before attention overwrites
        /// that same storage with the rotated values.
        func callAsFunction(
            _ x: MLXArray, queryRank: MLXArray, latent: MLXArray?, startPosition: Int,
            offset: Int, cache: DeepSeekV41Cache
        ) -> MLXArray {
            let (batch, length) = (x.dim(0), x.dim(1))
            let endPosition = startPosition + length

            if ownsKeys {
                if let latent {
                    publishKeys(latent: latent, startPosition: startPosition, cache: cache)
                }
                cache.indexKeys = cache.keys[layerIndex]
            }

            let (cos, sin) = rope.table(length: endPosition)
            let positions = MLXArray(startPosition ..< endPosition)
            var queries = wqB(DeepSeekV41FakeQuant.fp8UE8M0(queryRank))
                .reshaped(batch, length, heads, indexHeadDim)
            queries = DeepSeekV41IndexScores.applyRotary(
                queries,
                cos: take(cos, positions, axis: 0)[.newAxis, 0..., .newAxis, 0...],
                sin: take(sin, positions, axis: 0)[.newAxis, 0..., .newAxis, 0...],
                ropeDimensions: ropeHeadDim)
            queries = DeepSeekV41FakeQuant.fp4UE8M0(queries)

            let compressLength = endPosition / compressRatio
            let indexKeys = cache.indexKeys![0 ..< batch, 0 ..< compressLength]
            let weights = weightsProj(x) * (scale * pow(Float(heads), -0.5))
            let compressLengths =
                (MLXArray((startPosition + 1) ... endPosition)
                .floorDivide(compressRatio))[0..., .newAxis]
            let scores = DeepSeekV41IndexScores.scores(
                queries: queries, indexKeys: indexKeys, weights: weights,
                compressLengths: compressLengths,
                candidates: usesCandidates ? cache.candidates : nil)

            if isCandidateSource {
                cache.candidates = DeepSeekV41IndexScores.selectCandidateBlocks(
                    logits: scores, compressLengths: .perQuery(compressLengths),
                    topKBlocks: candidateTopKBlocks, blockSize: candidateBlockSize)
            }

            let topK = min(indexTopK, compressLength)
            let indices = MLX.sorted(
                argSort(-scores, axis: -1)[.ellipsis, ..<topK], axis: -1
            ).asType(.int32)
            return MLX.where(indices .< compressLengths, indices + offset, -1)
                .asType(.int32)
        }
    }

    static func yarnParameters(_ config: DeepSeekV41Configuration)
        -> DeepSeekV41IndexRope.YarnParameters
    {
        let scaling = config.ropeScaling ?? [:]
        return .init(
            factor: scaling["factor"]?.asFloat() ?? 1,
            betaFast: scaling["beta_fast"]?.asFloat() ?? 32,
            betaSlow: scaling["beta_slow"]?.asFloat() ?? 1,
            originalLength: scaling["original_max_position_embeddings"]?.asInt() ?? 65536)
    }

    /// Rotate strided positions (the compressor's latents) with explicit tables, paired
    /// the way the queries are.
    static func applyRopeAtPositions(
        _ x: MLXArray, positions: MLXArray, ropeDimensions: Int, theta: Float,
        yarn: DeepSeekV41IndexRope.YarnParameters
    ) -> MLXArray {
        let length = positions.size == 0 ? 1 : positions.max().item(Int32.self) + 1
        let rope = DeepSeekV41IndexRope(
            ropeDimensions: ropeDimensions, theta: theta, yarn: yarn)
        let (cos, sin) = rope.table(length: Int(length))
        let leading = Array(repeating: 1, count: x.ndim - positions.ndim - 1)
        let shape = leading + positions.shape + [cos.dim(-1)]
        return DeepSeekV41IndexScores.applyRotary(
            x,
            cos: take(cos, positions.reshaped(-1), axis: 0).reshaped(shape),
            sin: take(sin, positions.reshaped(-1), axis: 0).reshaped(shape),
            ropeDimensions: ropeDimensions)
    }
}

extension DeepSeekV41Language {
    /// Latent attention over the window KV plus the top-k compressed positions.
    ///
    /// The mode follows from the layer's role: kv sources own a compressor and an
    /// indexer, index-source non-owners share the KV and keys but rescore for
    /// themselves, the rest reuse both the KV and the selection, and ratio-0 layers are
    /// window-only. Every mode computes its own queries and its own sliding-window KV.
    /// Window and compressed KV stay in separate gathers, so no concatenation offset is
    /// needed, and the window is a fixed-size ring even when prefill arrives in chunks.
    final class Attention: Module {
        enum Mode {
            case local, full, reindex, reuse
        }

        let config: DeepSeekV41Configuration
        let layerIndex: Int
        let compressRatio: Int
        let isKVSource: Bool
        let isIndexSource: Bool
        let mode: Mode
        let heads: Int
        let headDim: Int
        let oGroups: Int
        let scale: Float
        let windowSize: Int
        let latentTheta: Float
        let latentYarn: DeepSeekV41IndexRope.YarnParameters

        @ModuleInfo(key: "wq_a") var wqA: Linear
        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "wq_b") var wqB: Linear
        @ModuleInfo(key: "wkv") var wkv: Linear
        @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm
        @ModuleInfo(key: "wo_a") var woA: DeepSeekV4Language.MultiLinear
        @ModuleInfo(key: "wo_b") var woB: Linear
        @ParameterInfo(key: "attn_sink") var attentionSink: MLXArray
        @ModuleInfo(key: "compressor") var compressor: Compressor?
        @ModuleInfo(key: "indexer") var indexer: Indexer?

        let rope: DeepSeekV4Language.RoPELayer

        init(_ config: DeepSeekV41Configuration, layerIndex: Int) {
            self.config = config
            self.layerIndex = layerIndex
            self.compressRatio = config.compressRatios[layerIndex]
            self.isKVSource = config.kvSourceLayerIds.contains(layerIndex)
            self.isIndexSource = config.indexSourceLayerIds.contains(layerIndex)
            if compressRatio == 0 {
                self.mode = .local
            } else if isKVSource {
                self.mode = .full
            } else if isIndexSource {
                self.mode = .reindex
            } else {
                self.mode = .reuse
            }
            self.heads = config.attentionHeads
            self.headDim = config.headDim
            self.oGroups = config.oGroups
            self.scale = pow(Float(config.headDim), -0.5)
            self.windowSize = config.slidingWindow
            self.latentTheta = config.compressRopeTheta
            self.latentYarn = DeepSeekV41Language.yarnParameters(config)

            _wqA.wrappedValue = Linear(config.hiddenSize, config.qLoraRank, bias: false)
            _qNorm.wrappedValue = RMSNorm(
                dimensions: config.qLoraRank, eps: config.rmsNormEps)
            _wqB.wrappedValue = Linear(
                config.qLoraRank, heads * headDim, bias: false)
            _wkv.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
            _kvNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
            _woA.wrappedValue = DeepSeekV4Language.MultiLinear(
                inputDims: heads * headDim / config.oGroups,
                outputDims: config.oLoraRank, groups: config.oGroups)
            _woB.wrappedValue = Linear(
                config.oGroups * config.oLoraRank, config.hiddenSize, bias: false)
            _attentionSink.wrappedValue = MLXArray.zeros([heads], dtype: .float32)

            self.rope = DeepSeekV4Language.RoPELayer(
                dims: config.qkRopeHeadDim,
                base: compressRatio > 0 ? config.compressRopeTheta : config.ropeTheta,
                scalingConfig: compressRatio > 0 ? config.ropeScaling : nil,
                maxPositionEmbeddings: config.maxPositionEmbeddings)

            _compressor.wrappedValue =
                isKVSource ? Compressor(config, layerIndex: layerIndex) : nil
            _indexer.wrappedValue =
                isIndexSource ? Indexer(config, layerIndex: layerIndex) : nil
            super.init()
        }

        /// Read the preceding window before overwriting its ring with this chunk.
        func windowPart(_ x: MLXArray, startPosition: Int, cache: DeepSeekV41Cache)
            -> (part: MLXArray, valid: MLXArray)
        {
            let (batch, length) = (x.dim(0), x.dim(1))
            let window = windowSize
            var kv = kvNorm(wkv(DeepSeekV41FakeQuant.fp8UE8M0(x)))
            kv = rope(kv.expandedDimensions(axis: 1), offset: startPosition)
                .reshaped(batch, length, headDim)
            kv = DeepSeekV41FakeQuant.fp8UE8M0(kv)

            var buffer = cache.window[layerIndex]
            if buffer == nil {
                buffer = MLXArray.zeros([batch, window, headDim], dtype: kv.dtype)
            } else if buffer!.dim(0) < batch {
                buffer = padded(
                    buffer!,
                    widths: [.init((0, batch - buffer!.dim(0))), .init((0, 0)), .init((0, 0))])
            }
            let base = max(0, startPosition - window + 1)
            var part = kv
            if startPosition > 0 {
                let history = take(
                    buffer![0 ..< batch], MLXArray(base ..< startPosition) % window, axis: 1)
                part = concatenated([history, kv], axis: 1)
            }
            let endPosition = startPosition + length
            let retained = min(length, window)
            buffer![0 ..< batch, MLXArray((endPosition - retained) ..< endPosition) % window] =
                kv[0..., (length - retained)...]
            cache.window[layerIndex] = buffer

            let positions = MLXArray(base ..< endPosition)
            let queries = MLXArray(startPosition ..< endPosition)
            let valid =
                (positions[.newAxis, 0...] .<= queries[0..., .newAxis])
                .&& (positions[.newAxis, 0...] .> (queries[0..., .newAxis] - window))
            return (part, valid[.newAxis, .newAxis])
        }

        /// The shared compressed KV slice plus this layer's top-k indices.
        func compressPart(
            _ x: MLXArray, queryRank: MLXArray, startPosition: Int, cache: DeepSeekV41Cache
        ) -> (pool: MLXArray, indices: MLXArray) {
            let (batch, length) = (x.dim(0), x.dim(1))
            var latent: MLXArray?
            if isKVSource {
                latent = compressor?(
                    x, startPosition: startPosition, state: cache.compressorState)
                if let latent {
                    let count = latent.dim(1)
                    let positions =
                        (MLXArray(0 ..< count) + startPosition / compressRatio)
                        * compressRatio
                    var poolKV = DeepSeekV41Language.applyRopeAtPositions(
                        latent, positions: positions,
                        ropeDimensions: config.qkRopeHeadDim, theta: latentTheta,
                        yarn: latentYarn)
                    poolKV = DeepSeekV41FakeQuant.fp4E4M3(poolKV)
                    cache.appendCompressed(layer: layerIndex, values: poolKV)
                }
                cache.compressKV = cache.compress[layerIndex]
            }

            let compressLength = (startPosition + length) / compressRatio
            let empty = (
                MLXArray.zeros([batch, 0, headDim], dtype: x.dtype),
                MLXArray.zeros([batch, length, 0], dtype: .int32)
            )
            guard compressLength > 0, let compressKV = cache.compressKV else { return empty }

            let pool = compressKV[0 ..< batch, 0 ..< compressLength]
            var indices: MLXArray?
            if isIndexSource {
                indices = indexer?(
                    x, queryRank: queryRank, latent: latent,
                    startPosition: startPosition, offset: 0, cache: cache)
                cache.topKIndices = indices
            } else {
                indices = cache.topKIndices
            }
            guard let indices else { return empty }
            return (pool, indices)
        }

        func callAsFunction(_ x: MLXArray, cache: DeepSeekV41Cache) -> MLXArray {
            let startPosition = cache.offset
            let (batch, length) = (x.dim(0), x.dim(1))
            let queryRank = qNorm(wqA(DeepSeekV41FakeQuant.fp8UE8M0(x)))
            var q = wqB(DeepSeekV41FakeQuant.fp8UE8M0(queryRank))
                .reshaped(batch, length, heads, headDim)
                .transposed(0, 2, 1, 3)
            q = rope(q, offset: startPosition)

            let (windowKV, windowMask) = windowPart(
                x, startPosition: startPosition, cache: cache)
            var out: MLXArray?
            if compressRatio > 0 {
                let (pool, indices) = compressPart(
                    x, queryRank: queryRank, startPosition: startPosition, cache: cache)
                if pool.dim(1) > 0 {
                    out = DeepSeekV4Language.Attention.sparsePooledAttention(
                        q: q.asType(.float32),
                        localKV: windowKV.expandedDimensions(axis: 1),
                        pooled: pool, topK: indices, localMask: windowMask,
                        pooledMask: (indices .!= -1).expandedDimensions(axis: 1),
                        scale: scale,
                        sinks: attentionSink.asType(.float32)
                    ).asType(q.dtype)
                }
            }
            if out == nil {
                let kv = windowKV.expandedDimensions(axis: 1).asType(q.dtype)
                out = MLXFast.scaledDotProductAttention(
                    queries: q, keys: kv, values: kv, scale: scale,
                    mask: .array(windowMask), sinks: attentionSink.asType(q.dtype))
            }

            var result = rope(out!, offset: startPosition, inverse: true)
            result = result.reshaped(batch, oGroups, -1, length, headDim)
            result = result.transposed(0, 1, 3, 2, 4).flattened(start: -2)
            result = woA(result)
            result = result.transposed(0, 2, 1, 3).flattened(start: -2)
            return woB(DeepSeekV41FakeQuant.fp8UE8M0(result))
        }
    }
}
