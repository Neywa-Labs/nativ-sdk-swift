import Foundation
import MLX
import MLXNN

enum Ernie55Attention {
    typealias Configuration = Ernie55Configuration

    class Layer: Module {
        func callAsFunction(_ x: MLXArray, cache: CacheList?) -> MLXArray {
            fatalError("An ERNIE attention layer requires an attention implementation")
        }
    }

    final class VirtualHeads: Module {
        @ParameterInfo(key: "U") var left: MLXArray
        @ParameterInfo(key: "V") var right: MLXArray
        let heads: Int

        init(heads: Int, rank: Int) {
            self.heads = heads
            _left.wrappedValue = .zeros([heads, rank])
            _right.wrappedValue = .zeros([heads, rank])
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let mix = MLXArray.eye(heads) + matmul(right.asType(.float32), left.asType(.float32).T)
            return matmul(mix.asType(x.dtype), x)
        }
    }

    final class Compressor: Module {
        @ModuleInfo var wkv: Linear
        @ModuleInfo var wgate: Linear
        @ParameterInfo var ape: MLXArray
        @ModuleInfo var norm: RMSNorm
        // Keep config-derived rotary frequencies out of the checkpoint state.
        let rope: (MLXArray, Int, Bool) -> MLXArray
        let ratio: Int
        let width: Int

        init(_ config: Configuration, ratio: Int) {
            self.ratio = ratio
            width = config.headDim
            _wkv.wrappedValue = Linear(config.hidden, width, bias: false)
            _wgate.wrappedValue = Linear(config.hidden, width, bias: false)
            _ape.wrappedValue = .zeros([ratio, width])
            _norm.wrappedValue = RMSNorm(dimensions: width, eps: config.rmsEps)
            rope =
                DeepSeekV4Language.RoPELayer(
                    dims: config.ropeDim, base: config.compressedTheta,
                    scalingConfig: config.ropeScaling, maxPositionEmbeddings: config.maxPositions,
                    freqScale: ratio
                ).callAsFunction
        }

        func callAsFunction(_ x: MLXArray, cache: DeepSeekV4PoolingCache?, offset: Int)
            -> MLXArray
        {
            let hidden = x.asType(.float32)
            let kv = matmul(hidden, wkv.weight.asType(.float32).T)
            let gate = matmul(hidden, wgate.weight.asType(.float32).T)
            let (readyKV, readyGate, base): (MLXArray, MLXArray, Int)
            if let cache {
                (readyKV, readyGate, base) = cache.accumulateWindows(
                    kv: kv, gate: gate, offset: offset)
            } else {
                let usable = kv.dim(1) / ratio * ratio
                (readyKV, readyGate, base) = (
                    kv[0..., ..<usable, 0...], gate[0..., ..<usable, 0...], offset
                )
            }

            let pooled: MLXArray
            if readyKV.dim(1) == 0 {
                pooled = .zeros([x.dim(0), 0, width], dtype: x.dtype)
            } else {
                let windows = readyKV.reshaped(x.dim(0), -1, ratio, width)
                let gates = readyGate.reshaped(windows.shape) + ape.asType(.float32)
                let value = norm((windows * softmax(gates, axis: -2, precise: true)).sum(axis: -2))
                pooled = rope(value.expandedDimensions(axis: 1), base, false)
                    .squeezed(axis: 1).asType(x.dtype)
            }
            return cache?.updateAndFetch(pooled) ?? pooled
        }
    }

    final class HCA: Layer {
        @ModuleInfo(key: "wq_a") var queryDown: Linear
        @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
        @ModuleInfo(key: "wq_b") var queryUp: Linear
        @ModuleInfo var wkv: Linear
        @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm
        @ModuleInfo(key: "wo_a") var outputGroups: DeepSeekV4Language.MultiLinear
        @ModuleInfo(key: "wo_b") var output: Linear
        @ModuleInfo(key: "o_gate") var outputGate: Linear
        @ModuleInfo var compressor: Compressor
        @ModuleInfo var vha: VirtualHeads
        @ParameterInfo(key: "attn_sink") var sinks: MLXArray
        // Keep config-derived rotary frequencies out of the checkpoint state.
        let rope: (MLXArray, Int, Bool) -> MLXArray
        let config: Configuration

        init(_ config: Configuration, ratio: Int) {
            self.config = config
            _queryDown.wrappedValue = Linear(config.hidden, config.queryRank, bias: false)
            _queryNorm.wrappedValue = RMSNorm(dimensions: config.queryRank, eps: config.rmsEps)
            _queryUp.wrappedValue = Linear(
                config.queryRank, config.heads * config.headDim, bias: false)
            _wkv.wrappedValue = Linear(config.hidden, config.headDim, bias: false)
            _kvNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsEps)
            _outputGroups.wrappedValue = DeepSeekV4Language.MultiLinear(
                inputDims: config.heads * config.headDim / config.outputGroups,
                outputDims: config.outputRank, groups: config.outputGroups)
            let groupedWidth = config.outputGroups * config.outputRank
            _output.wrappedValue = Linear(groupedWidth, config.hidden, bias: false)
            _outputGate.wrappedValue = Linear(config.queryRank, groupedWidth, bias: false)
            _compressor.wrappedValue = Compressor(config, ratio: ratio)
            _vha.wrappedValue = VirtualHeads(heads: config.heads, rank: config.virtualRank)
            _sinks.wrappedValue = .zeros([config.heads])
            rope =
                DeepSeekV4Language.RoPELayer(
                    dims: config.ropeDim, base: config.compressedTheta,
                    scalingConfig: config.ropeScaling, maxPositionEmbeddings: config.maxPositions
                )
                .callAsFunction
            super.init()
        }

        override func callAsFunction(_ x: MLXArray, cache: CacheList?) -> MLXArray {
            let (batch, length) = (x.dim(0), x.dim(1))
            let local = cache?[0]
            let pool = cache?[1] as? DeepSeekV4PoolingCache
            let offset = local?.offset ?? 0
            let qr = queryNorm(queryDown(x))
            let unrotated = queryUp(qr).reshaped(batch, length, config.heads, config.headDim)
            let queries = rope(
                MLXFast.rmsNorm(unrotated, weight: .mlxNone, eps: config.rmsEps)
                    .transposed(0, 2, 1, 3), offset, false)
            var kv = rope(kvNorm(wkv(x)).expandedDimensions(axis: 1), offset, false)
            if let local {
                (kv, _) = local.update(
                    keys: kv, values: .zeros([batch, 1, length, 0], dtype: kv.dtype))
                let storage = (local as? MaterializedKVCache)?.storage ?? local
                if let ring = storage as? RotatingKVCache,
                    let history = ring.logicalView(tail: config.window + length - 1)
                {
                    kv = history.0
                }
            }

            var mask: MLXArray?
            if length > 1 {
                mask = createCausalMask(
                    n: length, offset: kv.dim(2) - length, windowSize: config.window)
            }
            let pooled = compressor(x, cache: pool, offset: offset)
            if pooled.dim(1) > 0 {
                if let localMask = mask {
                    let positions = MLXArray((offset + 1) ... (offset + length)).floorDivide(
                        compressor.ratio)
                    let poolMask =
                        MLXArray(0 ..< pooled.dim(1))[.newAxis, 0...]
                        .< positions[0..., .newAxis]
                    mask = concatenated([localMask, poolMask], axis: -1)
                }
                kv = concatenated([kv, pooled.expandedDimensions(axis: 1)], axis: 2)
            }
            let attended = MLXFast.scaledDotProductAttention(
                queries: queries, keys: kv, values: kv, scale: 1 / sqrt(Float(config.headDim)),
                mask: mask.map { .array($0) } ?? .none, sinks: sinks.asType(queries.dtype))
            let mixed = vha(rope(attended, offset, true).transposed(0, 2, 1, 3))
            let grouped = mixed.reshaped(batch, length, config.outputGroups, -1)
                .transposed(0, 2, 1, 3)
            let projected = outputGroups(grouped).transposed(0, 2, 1, 3)
                .reshaped(batch, length, -1)
            return output(projected * sigmoid(outputGate(qr)))
        }
    }

    final class Indexer: Module {
        @ModuleInfo(key: "wq_b") var query: Linear
        @ModuleInfo var wk: Linear
        @ModuleInfo(key: "k_norm") var norm: LayerNorm
        @ModuleInfo(key: "weights_proj") var weights: Linear
        let config: Configuration
        let rope: RoPE

        init(_ config: Configuration) {
            self.config = config
            _query.wrappedValue = Linear(
                config.mlaQueryRank, config.indexHeads * config.indexDim, bias: false)
            _wk.wrappedValue = Linear(config.hidden, config.indexDim, bias: false)
            _norm.wrappedValue = LayerNorm(dimensions: config.indexDim, eps: 1e-6)
            _weights.wrappedValue = Linear(config.hidden, config.indexHeads, bias: false)
            rope = RoPE(dimensions: config.mlaRope, traditional: false, base: config.theta)
        }

        func callAsFunction(_ x: MLXArray, qr: MLXArray, mask: MLXArray?, cache: KVCache?)
            -> (mask: MLXArray?, gather: MLXArray?)
        {
            let (batch, length) = (x.dim(0), x.dim(1))
            let offset = cache?.offset ?? 0
            var keys = rope(norm(wk(x)).expandedDimensions(axis: 1), offset: offset)
            if let cache {
                (keys, _) = cache.update(
                    keys: keys, values: .zeros([batch, 1, length, 0], dtype: keys.dtype))
            }
            let keyLength = keys.dim(2)
            guard keyLength > config.indexTopK + config.window else { return (mask, nil) }

            let queries = rope(
                query(qr).reshaped(batch, length, config.indexHeads, config.indexDim)
                    .transposed(0, 2, 1, 3), offset: offset)
            let delta =
                MLXArray(0 ..< keyLength)[.newAxis, 0...]
                - MLXArray((keyLength - length) ..< keyLength)[0..., .newAxis]
            let recent = (delta .<= 0) .&& (delta .> -config.window)
            let older = delta .<= -config.window
            let valid = mask.map { older .&& $0 } ?? older
            let scores = maximum(matmul(queries, keys.swappedAxes(-1, -2)), 0).asType(.float32)
            let scale = 1 / sqrt(Float(config.indexHeads * config.indexDim))
            let coefficients = weights(x).asType(.float32) * scale
            let aggregate = (scores * coefficients.transposed(0, 2, 1).expandedDimensions(axis: -1))
                .sum(axis: 1, keepDims: true)
            let ranked = MLX.where(valid, aggregate, -Float.infinity)
            let ids = argPartition(-ranked, kth: config.indexTopK - 1, axis: -1)[
                .ellipsis, ..<config.indexTopK]
            let selected = putAlong(
                MLXArray.zeros(aggregate.shape, dtype: .bool), ids, values: MLXArray(true), axis: -1
            )
            let sparse = (selected .&& older) .|| recent
            let sparseMask = mask.map { sparse .&& $0 } ?? sparse
            guard length == 1 else { return (sparseMask, nil) }

            let suffix = broadcast(
                MLXArray((keyLength - config.window) ..< keyLength),
                to: [batch, 1, 1, config.window])
            return (sparseMask, concatenated([ids, suffix.asType(ids.dtype)], axis: -1))
        }
    }

    final class MLA: Layer {
        @ModuleInfo(key: "q_a_proj") var queryDown: Linear
        @ModuleInfo(key: "q_a_layernorm") var queryNorm: RMSNorm
        @ModuleInfo(key: "q_b_proj") var queryUp: Linear
        @ModuleInfo(key: "kv_a_proj_with_mqa") var kvDown: Linear
        @ModuleInfo(key: "kv_a_layernorm") var kvNorm: RMSNorm
        @ModuleInfo(key: "embed_q") var embedQuery: DeepSeekV4Language.MultiLinear
        @ModuleInfo(key: "unembed_out") var unembed: DeepSeekV4Language.MultiLinear
        @ModuleInfo(key: "wo_b") var output: Linear
        @ModuleInfo(key: "o_gate") var outputGate: Linear
        @ModuleInfo var vha: VirtualHeads
        @ModuleInfo var indexer: Indexer
        @ParameterInfo(key: "attn_sink") var sinks: MLXArray
        let config: Configuration
        let rope: RoPE
        let scale: Float

        init(_ config: Configuration) {
            self.config = config
            let queryWidth = config.mlaNope + config.mlaRope
            let outputWidth = config.mlaHeads * config.mlaValue
            scale = 1 / sqrt(Float(queryWidth))
            _queryDown.wrappedValue = Linear(config.hidden, config.mlaQueryRank, bias: false)
            _queryNorm.wrappedValue = RMSNorm(dimensions: config.mlaQueryRank, eps: config.rmsEps)
            _queryUp.wrappedValue = Linear(
                config.mlaQueryRank, config.mlaHeads * queryWidth, bias: false)
            _kvDown.wrappedValue = Linear(
                config.hidden, config.mlaLatent + config.mlaRope, bias: false)
            _kvNorm.wrappedValue = RMSNorm(dimensions: config.mlaLatent, eps: config.rmsEps)
            _embedQuery.wrappedValue = DeepSeekV4Language.MultiLinear(
                inputDims: config.mlaNope, outputDims: config.mlaLatent, groups: config.mlaHeads)
            _unembed.wrappedValue = DeepSeekV4Language.MultiLinear(
                inputDims: config.mlaLatent, outputDims: config.mlaValue, groups: config.mlaHeads)
            _output.wrappedValue = Linear(outputWidth, config.hidden, bias: false)
            _outputGate.wrappedValue = Linear(config.mlaQueryRank, outputWidth, bias: false)
            _vha.wrappedValue = VirtualHeads(heads: config.mlaHeads, rank: config.virtualRank)
            _indexer.wrappedValue = Indexer(config)
            _sinks.wrappedValue = .zeros([config.mlaHeads])
            rope = RoPE(dimensions: config.mlaRope, traditional: true, base: config.theta)
            super.init()
        }

        override func callAsFunction(_ x: MLXArray, cache: CacheList?) -> MLXArray {
            let (batch, length) = (x.dim(0), x.dim(1))
            let kvCache = cache?[0]
            let offset = kvCache?.offset ?? 0
            let mask = makeAttentionMask(n: length, cache: kvCache, returnArray: true).mask
            let qr = queryNorm(queryDown(x))
            let projected = queryUp(qr).reshaped(batch, length, config.mlaHeads, -1)
                .transposed(0, 2, 1, 3)
            var query = projected[.ellipsis, ..<config.mlaNope]
            let positionalQuery = rope(projected[.ellipsis, config.mlaNope...], offset: offset)
            let combined = kvDown(x)
            var latent = kvNorm(combined[.ellipsis, ..<config.mlaLatent]).expandedDimensions(
                axis: 1)
            var positions = rope(
                combined[.ellipsis, config.mlaLatent...].expandedDimensions(axis: 1), offset: offset
            )
            if let kvCache {
                (latent, positions) = kvCache.update(keys: latent, values: positions)
            }
            var (keep, gather) = indexer(x, qr: qr, mask: mask, cache: cache?[1])
            if let gather {
                let columns = gather.squeezed(axis: 2).expandedDimensions(axis: -1)
                latent = takeAlong(latent, columns, axis: 2)
                positions = takeAlong(positions, columns, axis: 2)
                keep = keep.map { takeAlong($0, gather, axis: -1) }
            }
            var bias = matmul(positionalQuery * scale, positions.swappedAxes(-1, -2))
            if let keep {
                bias = MLX.where(keep, bias, -Float.infinity)
            }
            let keys: MLXArray
            let values: MLXArray
            if length == 1 {
                query = embedQuery(query)
                keys = latent
                values = latent
            } else {
                keys = matmul(latent, embedQuery.weight)
                values = unembed(latent)
            }
            var attended = MLXFast.scaledDotProductAttention(
                queries: query, keys: keys, values: values, scale: scale,
                mask: .array(bias), sinks: sinks.asType(query.dtype))
            if length == 1 { attended = unembed(attended) }
            let mixed = vha(attended.transposed(0, 2, 1, 3)).reshaped(batch, length, -1)
            return output(mixed * sigmoid(outputGate(qr)))
        }
    }
}
