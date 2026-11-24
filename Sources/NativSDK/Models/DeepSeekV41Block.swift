import Foundation
import MLX
import MLXNN

extension DeepSeekV41Language {
    /// A block whose residual stream is `hc_mult` parallel copies.
    ///
    /// Attention and FFN each sit between a collapse of the copies and an expansion back
    /// out that mixes the residual in. The coefficients a sublayer computes are used by
    /// the *next* one: attention consumes the incoming mix and produces the FFN's, which
    /// produces the next block's.
    final class Block: Module {
        let layerIndex: Int
        let hcMultiplier: Int
        let sinkhornIterations: Int
        let hcEps: Float
        let normEps: Float

        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo(key: "ffn") var ffn: MoE
        @ModuleInfo(key: "engram") var engram: DeepSeekV41Engram?
        @ModuleInfo(key: "attn_norm") var attentionNorm: RMSNorm
        @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
        @ParameterInfo(key: "hc_attn_fn") var attentionFn: MLXArray
        @ParameterInfo(key: "hc_ffn_fn") var ffnFn: MLXArray
        @ParameterInfo(key: "hc_attn_base") var attentionBase: MLXArray
        @ParameterInfo(key: "hc_ffn_base") var ffnBase: MLXArray
        @ParameterInfo(key: "hc_attn_scale") var attentionScale: MLXArray
        @ParameterInfo(key: "hc_ffn_scale") var ffnScale: MLXArray

        init(
            _ config: DeepSeekV41Configuration, layerIndex: Int,
            engramLayout: DeepSeekV41EngramLayout? = nil
        ) {
            self.layerIndex = layerIndex
            self.hcMultiplier = config.hcMultiplier
            self.sinkhornIterations = config.hcSinkhornIterations
            self.hcEps = config.hcEps
            self.normEps = config.rmsNormEps

            _attention.wrappedValue = Attention(config, layerIndex: layerIndex)
            _ffn.wrappedValue = MoE(config)
            if let engramLayout, engramLayout.layerIds.contains(layerIndex) {
                _engram.wrappedValue = DeepSeekV41Engram(
                    layout: engramLayout, layerId: layerIndex,
                    hiddenSize: config.hiddenSize, hcMultiplier: config.hcMultiplier,
                    rmsNormEps: config.rmsNormEps)
            }
            _attentionNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _ffnNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)

            let mixDimensions = (2 + config.hcMultiplier) * config.hcMultiplier
            let hcDimensions = config.hcMultiplier * config.hiddenSize
            _attentionFn.wrappedValue = MLXArray.zeros(
                [mixDimensions, hcDimensions], dtype: .float32)
            _ffnFn.wrappedValue = MLXArray.zeros(
                [mixDimensions, hcDimensions], dtype: .float32)
            _attentionBase.wrappedValue = MLXArray.zeros([mixDimensions], dtype: .float32)
            _ffnBase.wrappedValue = MLXArray.zeros([mixDimensions], dtype: .float32)
            _attentionScale.wrappedValue = MLXArray.ones([3], dtype: .float32)
            _ffnScale.wrappedValue = MLXArray.ones([3], dtype: .float32)
            super.init()
        }

        func mixes(_ x: MLXArray, fn: MLXArray, scale: MLXArray, base: MLXArray)
            -> (pre: MLXArray, post: MLXArray, comb: MLXArray)
        {
            DeepSeekV41Language.hyperConnectionMixCoefficients(
                x, fn: fn, scale: scale, base: base, multiplier: hcMultiplier,
                sinkhornIterations: sinkhornIterations, eps: hcEps, normEps: normEps)
        }

        func callAsFunction(
            _ h: MLXArray, preMix: MLXArray, imageMask: MLXArray?, cache: DeepSeekV41Cache
        ) -> (MLXArray, MLXArray) {
            var residual = h
            let (attentionPre, attentionPost, attentionComb) = mixes(
                h, fn: attentionFn, scale: attentionScale, base: attentionBase)
            var x = DeepSeekV41Language.hyperConnectionPre(h, preMix: preMix)
            x = attentionNorm(x)
            x = attention(x, cache: cache)
            x = DeepSeekV41Language.hyperConnectionExpand(
                x, residual: residual, post: attentionPost, comb: attentionComb)

            residual = x
            let (ffnPre, ffnPost, ffnComb) = mixes(
                x, fn: ffnFn, scale: ffnScale, base: ffnBase)
            x = DeepSeekV41Language.hyperConnectionPre(x, preMix: attentionPre)
            x = ffnNorm(x)
            x = ffn(x, imageMask: imageMask)
            x = DeepSeekV41Language.hyperConnectionExpand(
                x, residual: residual, post: ffnPost, comb: ffnComb)
            return (x, ffnPre)
        }
    }

    /// Embed, expand to the hc copies, run the blocks, collapse, project to logits.
    final class LanguageModel: Module {
        let config: DeepSeekV41Configuration
        let layout: DeepSeekV41EngramLayout?
        var engramHash: DeepSeekV41NgramHashState?

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        let layers: [Block]
        @ModuleInfo(key: "norm") var norm: RMSNorm
        @ModuleInfo(key: "head") var head: ParallelHead

        init(_ config: DeepSeekV41Configuration) {
            self.config = config
            self.layout = DeepSeekV41EngramLayout(
                layerIds: config.engramLayerIds,
                maxNgramSize: config.engramMaxNgramSize, heads: config.engramHeads,
                headDim: config.engramHeadDim,
                vocabularySize: config.engramVocabularySize,
                numEmbeddings: config.engramNumEmbeddings)
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            let engramLayout = layout
            self.layers = (0 ..< config.hiddenLayers).map {
                Block(config, layerIndex: $0, engramLayout: engramLayout)
            }
            _norm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _head.wrappedValue = ParallelHead(config)
            super.init()
        }

        func newCache() -> [KVCache] {
            [DeepSeekV41Cache(layers: layers.count)]
        }

        /// `inputIds` is read even when `inputsEmbeds` is supplied: the Engram routes on
        /// the token ids, so without them the tables contribute nothing and a chunked
        /// prompt encodes differently from an unchunked one.
        func callAsFunction(
            _ inputIds: MLXArray?, inputsEmbeds: MLXArray? = nil,
            cache: DeepSeekV41Cache? = nil, imageMask: MLXArray? = nil,
            engramHashes: MLXArray? = nil
        ) -> MLXArray {
            let entry = cache ?? DeepSeekV41Cache(layers: layers.count)
            var mask = imageMask
            if mask == nil, let inputIds {
                mask = inputIds .== config.imageTokenId
            }
            let engramMask = mask.map { logicalNot($0) }

            var h = inputsEmbeds ?? embedTokens(inputIds!)
            let (batch, length) = (h.dim(0), h.dim(1))
            h = broadcast(
                h[.ellipsis, .newAxis, 0...],
                to: [batch, length, config.hcMultiplier, config.hiddenSize])

            var hashes = engramHashes
            if hashes == nil, let inputIds, let engramHash {
                hashes = engramHash(
                    inputIds, startPosition: entry.offset, history: &entry.engram,
                    tokenMask: engramMask)
            }

            var preMix = DeepSeekV41Language.identityPreMix(
                batch: batch, length: length, multiplier: config.hcMultiplier)
            for layer in layers {
                if let engram = layer.engram, let hashes {
                    h = engram(
                        h, hashIds: hashes[0..., 0..., engram.layerHashIndex, 0...],
                        tokenMask: engramMask)
                }
                (h, preMix) = layer(h, preMix: preMix, imageMask: mask, cache: entry)
            }
            h = DeepSeekV41Language.hyperConnectionPre(h, preMix: preMix)
            let logits = head(norm(h))
            entry.advance(length)
            return logits
        }
    }
}
