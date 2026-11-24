import Foundation
import MLX
import MLXNN

public struct DeepSeekV4Configuration: Codable, Sendable {
    public let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let moeIntermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let nSharedExperts: Int
    public let nRoutedExperts: Int
    public let routedScalingFactor: Float
    public let qLoraRank: Int
    public let qkRopeHeadDim: Int
    public let numExpertsPerToken: Int
    public let normTopkProb: Bool
    public let hiddenActivation: String
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let ropeScaling: [String: StringOrNumber]?
    public let attentionBias: Bool
    public let headDim: Int
    public let scoringFunction: String
    public let compressRatios: [Int]
    public let compressRopeTheta: Float
    public let hcMult: Int
    public let hcSinkhornIterations: Int
    public let hcEps: Float
    public let numHashLayers: Int
    public let swigluLimit: Float
    public let slidingWindow: Int
    public let outputGroups: Int
    public let outputLoraRank: Int
    public let indexHeads: Int
    public let indexHeadDim: Int
    public let indexTopK: Int
    public let indexBlock: Int
    public let indexKeep: Int
    public let tieWordEmbeddings: Bool
    public let bosTokenID: Int?
    public let eosTokenID: IntOrIntArray?
    public let padTokenID: Int?
    public let topKMethod: String

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case numExpertsPerToken = "num_experts_per_tok"
        case normTopkProb = "norm_topk_prob"
        case hiddenActivation = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case headDim = "head_dim"
        case scoringFunction = "scoring_func"
        case compressRatios = "compress_ratios"
        case compressRopeTheta = "compress_rope_theta"
        case hcMult = "hc_mult"
        case hcSinkhornIterations = "hc_sinkhorn_iters"
        case hcEps = "hc_eps"
        case numHashLayers = "num_hash_layers"
        case swigluLimit = "swiglu_limit"
        case slidingWindow = "sliding_window"
        case outputGroups = "o_groups"
        case outputLoraRank = "o_lora_rank"
        case indexHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexTopK = "index_topk"
        case indexBlock = "index_block"
        case indexKeep = "index_keep"
        case tieWordEmbeddings = "tie_word_embeddings"
        case bosTokenID = "bos_token_id"
        case eosTokenID = "eos_token_id"
        case padTokenID = "pad_token_id"
        case topKMethod = "topk_method"
    }

    public init(
        modelType: String = "deepseek_v4",
        vocabularySize: Int = 129_280,
        hiddenSize: Int = 4096,
        intermediateSize: Int = 18432,
        moeIntermediateSize: Int = 2048,
        hiddenLayers: Int = 43,
        attentionHeads: Int = 64,
        kvHeads: Int = 1,
        nSharedExperts: Int = 1,
        nRoutedExperts: Int = 256,
        routedScalingFactor: Float = 1.5,
        qLoraRank: Int = 1024,
        qkRopeHeadDim: Int = 64,
        numExpertsPerToken: Int = 6,
        normTopkProb: Bool = true,
        hiddenActivation: String = "silu",
        maxPositionEmbeddings: Int = 1_048_576,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 10000,
        ropeScaling: [String: StringOrNumber]? = nil,
        attentionBias: Bool = false,
        headDim: Int = 512,
        scoringFunction: String = "sqrtsoftplus",
        compressRatios: [Int]? = nil,
        compressRopeTheta: Float = 160_000,
        hcMult: Int = 4,
        hcSinkhornIterations: Int = 20,
        hcEps: Float = 1e-6,
        numHashLayers: Int = 3,
        swigluLimit: Float = 10,
        slidingWindow: Int = 128,
        outputGroups: Int = 8,
        outputLoraRank: Int = 1024,
        indexHeads: Int = 64,
        indexHeadDim: Int = 128,
        indexTopK: Int = 512,
        indexBlock: Int = 0,
        indexKeep: Int = 0,
        tieWordEmbeddings: Bool = false,
        bosTokenID: Int? = nil,
        eosTokenID: IntOrIntArray? = nil,
        padTokenID: Int? = nil,
        topKMethod: String = "noaux_tc",
    ) {
        self.modelType = modelType
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.nSharedExperts = nSharedExperts
        self.nRoutedExperts = nRoutedExperts
        self.routedScalingFactor = routedScalingFactor
        self.qLoraRank = qLoraRank
        self.qkRopeHeadDim = qkRopeHeadDim
        self.numExpertsPerToken = numExpertsPerToken
        self.normTopkProb = normTopkProb
        self.hiddenActivation = hiddenActivation
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.attentionBias = attentionBias
        self.headDim = headDim
        self.scoringFunction = scoringFunction
        self.compressRatios = Self.normalizedCompressRatios(
            compressRatios,
            hiddenLayers: hiddenLayers,
        )
        self.compressRopeTheta = compressRopeTheta
        self.hcMult = hcMult
        self.hcSinkhornIterations = hcSinkhornIterations
        self.hcEps = hcEps
        self.numHashLayers = numHashLayers
        self.swigluLimit = swigluLimit
        self.slidingWindow = slidingWindow
        self.outputGroups = outputGroups
        self.outputLoraRank = outputLoraRank
        self.indexHeads = indexHeads
        self.indexHeadDim = indexHeadDim
        self.indexTopK = indexTopK
        self.indexBlock = indexBlock
        self.indexKeep = indexKeep
        self.tieWordEmbeddings = tieWordEmbeddings
        self.bosTokenID = bosTokenID
        self.eosTokenID = eosTokenID
        self.padTokenID = padTokenID
        self.topKMethod = topKMethod
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 43
        try self.init(
            modelType: c.decodeIfPresent(String.self, forKey: .modelType) ?? "deepseek_v4",
            vocabularySize:
                c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 129_280,
            hiddenSize: c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096,
            intermediateSize:
                c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 18432,
            moeIntermediateSize:
                c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 2048,
            hiddenLayers: hiddenLayers,
            attentionHeads:
                c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 64,
            kvHeads: c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 1,
            nSharedExperts:
                c.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 1,
            nRoutedExperts:
                c.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 256,
            routedScalingFactor:
                c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.5,
            qLoraRank: c.decodeIfPresent(Int.self, forKey: .qLoraRank) ?? 1024,
            qkRopeHeadDim:
                c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64,
            numExpertsPerToken:
                c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 6,
            normTopkProb:
                c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true,
            hiddenActivation:
                c.decodeIfPresent(String.self, forKey: .hiddenActivation) ?? "silu",
            maxPositionEmbeddings:
                c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 1_048_576,
            rmsNormEps: c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6,
            ropeTheta: c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000,
            ropeScaling:
                c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling),
            attentionBias:
                c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false,
            headDim: c.decodeIfPresent(Int.self, forKey: .headDim) ?? 512,
            scoringFunction:
                c.decodeIfPresent(String.self, forKey: .scoringFunction) ?? "sqrtsoftplus",
            compressRatios:
                c.decodeIfPresent([Int].self, forKey: .compressRatios),
            compressRopeTheta:
                c.decodeIfPresent(Float.self, forKey: .compressRopeTheta) ?? 160_000,
            hcMult: c.decodeIfPresent(Int.self, forKey: .hcMult) ?? 4,
            hcSinkhornIterations:
                c.decodeIfPresent(Int.self, forKey: .hcSinkhornIterations) ?? 20,
            hcEps: c.decodeIfPresent(Float.self, forKey: .hcEps) ?? 1e-6,
            numHashLayers:
                c.decodeIfPresent(Int.self, forKey: .numHashLayers) ?? 3,
            swigluLimit:
                c.decodeIfPresent(Float.self, forKey: .swigluLimit) ?? 10,
            slidingWindow:
                c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 128,
            outputGroups:
                c.decodeIfPresent(Int.self, forKey: .outputGroups) ?? 8,
            outputLoraRank:
                c.decodeIfPresent(Int.self, forKey: .outputLoraRank) ?? 1024,
            indexHeads: c.decodeIfPresent(Int.self, forKey: .indexHeads) ?? 64,
            indexHeadDim:
                c.decodeIfPresent(Int.self, forKey: .indexHeadDim) ?? 128,
            indexTopK: c.decodeIfPresent(Int.self, forKey: .indexTopK) ?? 512,
            indexBlock: c.decodeIfPresent(Int.self, forKey: .indexBlock) ?? 0,
            indexKeep: c.decodeIfPresent(Int.self, forKey: .indexKeep) ?? 0,
            tieWordEmbeddings:
                c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false,
            bosTokenID: c.decodeIfPresent(Int.self, forKey: .bosTokenID),
            eosTokenID: c.decodeIfPresent(IntOrIntArray.self, forKey: .eosTokenID),
            padTokenID: c.decodeIfPresent(Int.self, forKey: .padTokenID),
            topKMethod: c.decodeIfPresent(String.self, forKey: .topKMethod) ?? "noaux_tc",
        )
    }

    private static func normalizedCompressRatios(_ ratios: [Int]?, hiddenLayers: Int) -> [Int] {
        var result = ratios ?? defaultCompressRatios(hiddenLayers: hiddenLayers)
        result = Array(result.prefix(hiddenLayers))
        if result.count < hiddenLayers {
            result.append(contentsOf: Array(repeating: 0, count: hiddenLayers - result.count))
        }
        return result
    }

    private static func defaultCompressRatios(hiddenLayers: Int) -> [Int] {
        guard hiddenLayers > 0 else {
            return []
        }
        if hiddenLayers == 1 {
            return [0]
        }
        var ratios = [0]
        for i in 0 ..< max(hiddenLayers - 2, 0) {
            ratios.append(i % 2 == 0 ? 128 : 4)
        }
        ratios.append(0)
        return Array(ratios.prefix(hiddenLayers))
    }
}

final class DeepSeekV4PoolingCache: KVCache {
    let ratio: Int
    var offset = 0
    var maxSize: Int? {
        nil
    }

    var bufKV: MLXArray?
    var bufGate: MLXArray?
    var remainder = 0
    var pooled: MLXArray?

    init(ratio: Int) {
        self.ratio = ratio
    }

    func innerState() -> [MLXArray] {
        state
    }

    func update(keys _: MLXArray, values _: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("DeepSeekV4PoolingCache does not store standard KV pairs.")
    }

    func accumulateWindows(kv: MLXArray, gate: MLXArray, offset: Int)
        -> (MLXArray, MLXArray, Int)
    {
        let (batch, length, kvDim) = (kv.dim(0), kv.dim(1), kv.dim(2))
        let gateDim = gate.dim(2)
        if bufKV == nil {
            bufKV = zeros([batch, ratio, kvDim], dtype: kv.dtype)
            bufGate = zeros([batch, ratio, gateDim], dtype: gate.dtype)
        }

        if length > 1 {
            let total = length + remainder
            let usable = (total / ratio) * ratio
            let newRemainder = total % ratio
            let readyKV: MLXArray
            let readyGate: MLXArray
            let readyBase: Int
            if usable > 0 {
                readyKV = concatenated(
                    [bufKV![0..., ..<remainder, 0...], kv[0..., ..<(usable - remainder), 0...]],
                    axis: 1,
                )
                readyGate = concatenated(
                    [
                        bufGate![0..., ..<remainder, 0...],
                        gate[0..., ..<(usable - remainder), 0...],
                    ],
                    axis: 1,
                )
                readyBase = offset - remainder
                remainder = 0
            } else {
                readyKV = zeros([batch, 0, kvDim], dtype: kv.dtype)
                readyGate = zeros([batch, 0, gateDim], dtype: gate.dtype)
                readyBase = 0
            }

            if newRemainder > 0 {
                // If no window finished, append the entire input chunk to the buffered prefix.
                let start = max(0, length - newRemainder)
                bufKV![0..., remainder ..< newRemainder, 0...] = kv[0..., start..., 0...]
                bufGate![0..., remainder ..< newRemainder, 0...] = gate[0..., start..., 0...]
            }
            remainder = newRemainder
            return (readyKV, readyGate, readyBase)
        }

        bufKV![0..., remainder ..< (remainder + 1), 0...] = kv
        bufGate![0..., remainder ..< (remainder + 1), 0...] = gate
        remainder = (remainder + 1) % ratio
        if remainder == 0 {
            return (bufKV!, bufGate!, offset - ratio + 1)
        }
        return (
            zeros([batch, 0, kvDim], dtype: kv.dtype),
            zeros([batch, 0, gateDim], dtype: gate.dtype),
            0,
        )
    }

    func updateAndFetch(_ pooledTokens: MLXArray) -> MLXArray {
        if pooledTokens.dim(1) == 0 {
            if let pooled {
                return pooled
            }
            return zeros([pooledTokens.dim(0), 0, pooledTokens.dim(-1)], dtype: pooledTokens.dtype)
        }
        if let pooled {
            self.pooled = concatenated([pooled, pooledTokens], axis: 1)
        } else {
            pooled = pooledTokens
        }
        offset = pooled?.dim(1) ?? 0
        return pooled!
    }

    func makePooledMask(length: Int, offset: Int) -> MLXArray? {
        guard let pooled, length > 1 else {
            return nil
        }
        let poolIndex = MLXArray(0 ..< pooled.dim(1))
        let queryIndex = MLXArray((offset + 1) ..< (offset + length + 1))
        return poolIndex[.newAxis, 0...] .< queryIndex[0..., .newAxis].floorDivide(ratio)
    }

    var state: [MLXArray] {
        get {
            var result: [MLXArray] = []
            if remainder > 0, let bufKV, let bufGate {
                result.append(bufKV[0..., ..<remainder, 0...])
                result.append(bufGate[0..., ..<remainder, 0...])
            }
            if let pooled {
                result.append(pooled)
            }
            return result
        }
        set {
            pooled = newValue.last
            offset = pooled?.dim(1) ?? 0
        }
    }

    var metaState: [String] {
        get { ["\(ratio)", "\(remainder)"] }
        set {
            if let first = newValue.first.flatMap(Int.init) {
                precondition(first == ratio, "Cannot restore PoolingCache with different ratio.")
            }
            if newValue.count > 1, let value = Int(newValue[1]) {
                remainder = value
            }
        }
    }

    var isTrimmable: Bool {
        false
    }

    @discardableResult
    func trim(_: Int) -> Int {
        0
    }

    func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool,
    )
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        makeAttentionMask(n: n, cache: nil, windowSize: windowSize, returnArray: returnArray)
    }

    func copy() -> any KVCache {
        let new = DeepSeekV4PoolingCache(ratio: ratio)
        new.remainder = remainder
        new.bufKV = bufKV.map { $0[.ellipsis] }
        new.bufGate = bufGate.map { $0[.ellipsis] }
        new.pooled = pooled.map { $0[.ellipsis] }
        new.offset = offset
        return new
    }
}

extension DeepSeekV4PoolingCache: CacheCapabilityProviding {
    public var capabilities: CacheCapabilities { [.snapshot, .prefixRewind] }
}

enum DeepSeekV4Language {
    final class RoPELayer: Module {
        let dims: Int
        let freqScale: Int
        let frequencies: MLXArray

        init(
            dims: Int,
            base: Float,
            scalingConfig: [String: StringOrNumber]?,
            maxPositionEmbeddings: Int,
            freqScale: Int = 1,
        ) {
            self.dims = dims
            self.freqScale = freqScale
            var inverseFrequency =
                1.0 / pow(base, arange(0, dims, step: 2, dtype: .float32) / Float(dims))
            let ropeType =
                scalingConfig?["type"]?.asString()
                ?? scalingConfig?["rope_type"]?.asString()

            if ropeType == "yarn" || ropeType == "deepseek_yarn" {
                let factor = scalingConfig?["factor"]?.asFloat() ?? 1
                let originalMax =
                    scalingConfig?["original_max_position_embeddings"]?.asInt()
                    ?? maxPositionEmbeddings
                let betaFast = scalingConfig?["beta_fast"]?.asFloat() ?? 32
                let betaSlow = scalingConfig?["beta_slow"]?.asFloat() ?? 1

                func correctionDim(_ rotations: Float) -> Float {
                    Float(dims)
                        * log(Float(originalMax) / (rotations * 2 * Float.pi))
                        / (2 * log(base))
                }

                let low = max(floor(correctionDim(betaFast)), 0)
                var high = min(ceil(correctionDim(betaSlow)), Float(dims - 1))
                if low == high {
                    high += 0.001
                }
                let ramp = (arange(0, dims / 2, dtype: .float32) - low) / (high - low)
                let smooth = 1 - clip(ramp, min: Float32(0), max: Float32(1))
                inverseFrequency =
                    inverseFrequency / factor * (1 - smooth)
                    + inverseFrequency * smooth
            }
            frequencies = 1.0 / inverseFrequency
            super.init()
        }

        func callAsFunction(_ x: MLXArray, offset: Int, inverse: Bool = false) -> MLXArray {
            var freqs = frequencies
            if freqScale != 1 {
                freqs = freqs / freqScale
            }
            if inverse {
                freqs = -freqs
            }
            let headDim = x.dim(-1)
            let nopePairs = (headDim - dims) / 2
            if nopePairs > 0 {
                freqs = concatenated([
                    full([nopePairs], values: MLXArray(Float.infinity), dtype: .float32),
                    freqs,
                ])
            }
            let adjustedOffset = freqScale == 1 ? offset : offset / freqScale
            return MLXFast.RoPE(
                x,
                dimensions: headDim,
                traditional: true,
                base: nil,
                scale: 1.0,
                offset: adjustedOffset,
                freqs: freqs,
            )
        }
    }

    final class MultiLinear: Module, Quantizable {
        @ModuleInfo var weight: MLXArray
        let inputDims: Int
        let outputDims: Int
        let groups: Int

        init(inputDims: Int, outputDims: Int, groups: Int) {
            self.inputDims = inputDims
            self.outputDims = outputDims
            self.groups = groups
            let scale = sqrt(1.0 / Float(inputDims))
            _weight.wrappedValue = MLXRandom.uniform(
                low: -scale,
                high: scale,
                [groups, outputDims, inputDims],
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            x.matmul(weight.swappedAxes(-1, -2))
        }

        func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
            QuantizedMultiLinear(
                weight: weight,
                groupSize: groupSize,
                bits: bits,
                mode: mode,
            )
        }
    }

    final class QuantizedMultiLinear: Module, Quantized {
        let groupSize: Int
        let bits: Int
        let mode: QuantizationMode

        @ModuleInfo var weight: MLXArray
        @ModuleInfo var scales: MLXArray
        @ModuleInfo var biases: MLXArray?

        init(weight: MLXArray, groupSize: Int, bits: Int, mode: QuantizationMode) {
            self.groupSize = groupSize
            self.bits = bits
            self.mode = mode
            let (quantizedWeight, scales, biases) =
                MLX.quantized(weight, groupSize: groupSize, bits: bits, mode: mode)
            _weight.wrappedValue = quantizedWeight
            _scales.wrappedValue = scales
            _biases.wrappedValue = biases
            super.init()
            freeze()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            quantizedMM(
                x,
                weight,
                scales: scales,
                biases: biases,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: mode,
            )
        }
    }

    final class HyperConnection: Module {
        let config: DeepSeekV4Configuration
        let mixDimensions: Int

        @ModuleInfo var fn: MLXArray
        @ModuleInfo var base: MLXArray
        @ModuleInfo var scale: MLXArray

        init(_ config: DeepSeekV4Configuration) {
            self.config = config
            mixDimensions = (2 + config.hcMult) * config.hcMult
            _fn.wrappedValue = zeros([mixDimensions, config.hcMult * config.hiddenSize])
            _base.wrappedValue = zeros([mixDimensions])
            _scale.wrappedValue = ones([3])
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
            let y = x.asType(.float32)
            let z = MLXFast.rmsNorm(
                y.flattened(start: -2, end: -1),
                weight: MLXArray.mlxNone,
                eps: config.rmsNormEps,
            )
            let mixes = z.matmul(fn.T)
            let h = config.hcMult
            let pre =
                sigmoid(
                    mixes[.ellipsis, ..<h] * scale[0] + base[..<h],
                ) + config.hcEps
            let post =
                2
                * sigmoid(
                    mixes[.ellipsis, h ..< (2 * h)] * scale[1] + base[h ..< (2 * h)],
                )
            var comb = mixes[.ellipsis, (2 * h)...].reshaped(mixes.shape.dropLast() + [h, h])
            comb = comb * scale[2] + base[(2 * h)...].reshaped(h, h)
            comb = softmax(comb, axis: -1, precise: true) + config.hcEps
            comb = comb / (comb.sum(axis: -2, keepDims: true) + config.hcEps)
            for _ in 0 ..< max(config.hcSinkhornIterations - 1, 0) {
                comb = comb / (comb.sum(axis: -1, keepDims: true) + config.hcEps)
                comb = comb / (comb.sum(axis: -2, keepDims: true) + config.hcEps)
            }
            let collapsed = (pre[.ellipsis, .newAxis] * y).sum(axis: 2).asType(x.dtype)
            return (collapsed, post, comb)
        }
    }

    static func hcExpand(
        _ x: MLXArray,
        residual: MLXArray,
        post: MLXArray,
        comb: MLXArray,
    )
        -> MLXArray
    {
        let y =
            post[.ellipsis, .newAxis] * x[0..., 0..., .newAxis, 0...].asType(.float32)
            + comb.swappedAxes(-1, -2).matmul(residual.asType(.float32))
        return y.asType(x.dtype)
    }

    final class HyperHead: Module {
        let rmsEps: Float
        let hcEps: Float

        @ModuleInfo var fn: MLXArray
        @ModuleInfo var base: MLXArray
        @ModuleInfo var scale: MLXArray

        convenience init(_ config: DeepSeekV4Configuration) {
            self.init(
                hidden: config.hiddenSize, streams: config.hcMult,
                rmsEps: config.rmsNormEps, hcEps: config.hcEps)
        }

        init(hidden: Int, streams: Int, rmsEps: Float, hcEps: Float) {
            self.rmsEps = rmsEps
            self.hcEps = hcEps
            _fn.wrappedValue = zeros([streams, streams * hidden])
            _base.wrappedValue = zeros([streams])
            _scale.wrappedValue = ones([1])
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let y = x.asType(.float32)
            let z = MLXFast.rmsNorm(
                y.flattened(start: -2, end: -1),
                weight: MLXArray.mlxNone,
                eps: rmsEps,
            )
            let mixes = z.matmul(fn.T)
            let pre = sigmoid(mixes * scale + base) + hcEps
            return (pre[.ellipsis, .newAxis] * y).sum(axis: 2).asType(x.dtype)
        }
    }

    final class MLP: Module, UnaryLayer {
        let limit: Float

        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        init(_ config: DeepSeekV4Configuration, intermediateSize: Int? = nil) {
            let hidden = intermediateSize ?? config.intermediateSize
            limit = config.swigluLimit
            _gateProj.wrappedValue = Linear(config.hiddenSize, hidden, bias: false)
            _upProj.wrappedValue = Linear(config.hiddenSize, hidden, bias: false)
            _downProj.wrappedValue = Linear(hidden, config.hiddenSize, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(limitedSwiGLU(gateProj(x), upProj(x), limit: limit))
        }
    }

    static func limitedSwiGLU(_ gate: MLXArray, _ up: MLXArray, limit: Float) -> MLXArray {
        if limit > 0 {
            return silu(minimum(gate, limit)) * clip(up, min: -limit, max: limit)
        }
        return silu(gate) * up
    }

    final class SwitchMLP: Module {
        let limit: Float
        @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
        @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
        @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

        init(_ config: DeepSeekV4Configuration) {
            limit = config.swigluLimit
            _gateProj.wrappedValue = SwitchLinear(
                inputDims: config.hiddenSize,
                outputDims: config.moeIntermediateSize,
                numExperts: config.nRoutedExperts,
                bias: false,
            )
            _upProj.wrappedValue = SwitchLinear(
                inputDims: config.hiddenSize,
                outputDims: config.moeIntermediateSize,
                numExperts: config.nRoutedExperts,
                bias: false,
            )
            _downProj.wrappedValue = SwitchLinear(
                inputDims: config.moeIntermediateSize,
                outputDims: config.hiddenSize,
                numExperts: config.nRoutedExperts,
                bias: false,
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
            var x = expandedDimensions(x, axes: [-2, -3])
            let doSort = indices.size >= 64
            var idx = indices
            var inverseOrder = MLXArray()
            if doSort {
                (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
            }
            let up = upProj(x, idx, sortedIndices: doSort)
            let gate = gateProj(x, idx, sortedIndices: doSort)
            var output = downProj(limitedSwiGLU(gate, up, limit: limit), idx, sortedIndices: doSort)
            if doSort {
                output = scatterUnsort(x: output, invOrder: inverseOrder, shape: indices.shape)
            }
            return squeezed(output, axis: -2)
        }
    }

    final class MoEGate: Module {
        let config: DeepSeekV4Configuration
        let isHash: Bool

        @ModuleInfo var weight: MLXArray
        @ModuleInfo(key: "tid2eid") var tokenToExpert: MLXArray?
        @ModuleInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray?

        init(_ config: DeepSeekV4Configuration, layerIndex: Int) {
            self.config = config
            isHash = layerIndex < config.numHashLayers
            _weight.wrappedValue = zeros([config.nRoutedExperts, config.hiddenSize])
            if isHash {
                _tokenToExpert.wrappedValue = zeros(
                    [config.vocabularySize, config.numExpertsPerToken],
                    dtype: .int32,
                )
            } else {
                _eScoreCorrectionBias.wrappedValue = zeros([config.nRoutedExperts])
            }
            super.init()
        }

        func callAsFunction(_ x: MLXArray, inputIDs: MLXArray) -> (MLXArray, MLXArray) {
            let logits = x.matmul(weight.T).asType(.float32)
            let scores = score(logits)
            let indices: MLXArray
            if isHash {
                indices = tokenToExpert![inputIDs].asType(.int32)
            } else {
                let biased = scores + eScoreCorrectionBias!.asType(scores.dtype)
                let topK = min(config.numExpertsPerToken, config.nRoutedExperts)
                indices = argPartition(-biased, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
                    .asType(.int32)
            }
            var weights = takeAlong(scores, indices, axis: -1)
            if config.scoringFunction != "softmax", config.normTopkProb {
                weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
            }
            weights = weights * config.routedScalingFactor
            return (indices, weights.asType(x.dtype))
        }

        private func score(_ logits: MLXArray) -> MLXArray {
            switch config.scoringFunction {
            case "softmax":
                softmax(logits, axis: -1, precise: true)
            case "sigmoid":
                sigmoid(logits)
            case "sqrtsoftplus":
                sqrt(softplus(logits))
            default:
                preconditionFailure(
                    "Unsupported DeepSeek V4 scoring function: \(config.scoringFunction)")
            }
        }
    }

    final class MoE: Module {
        @ModuleInfo var gate: MoEGate
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchMLP
        @ModuleInfo(key: "shared_experts") var sharedExperts: MLP

        init(_ config: DeepSeekV4Configuration, layerIndex: Int) {
            _gate.wrappedValue = MoEGate(config, layerIndex: layerIndex)
            _switchMLP.wrappedValue = SwitchMLP(config)
            _sharedExperts.wrappedValue = MLP(
                config,
                intermediateSize: config.moeIntermediateSize * config.nSharedExperts,
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray, inputIDs: MLXArray) -> MLXArray {
            let (indices, scores) = gate(x, inputIDs: inputIDs)
            var y = switchMLP(x, indices)
            y = (y * scores[.ellipsis, .newAxis].asType(y.dtype)).sum(axis: -2)
            return y + sharedExperts(x)
        }
    }

    final class Compressor: Module {
        let config: DeepSeekV4Configuration
        let compressRatio: Int
        let headDim: Int
        let overlap: Bool
        let outputDim: Int

        @ModuleInfo var wkv: Linear
        @ModuleInfo var wgate: Linear
        @ModuleInfo var ape: MLXArray
        @ModuleInfo var norm: RMSNorm
        let rope: RoPELayer

        init(_ config: DeepSeekV4Configuration, compressRatio: Int, headDim: Int) {
            self.config = config
            self.compressRatio = compressRatio
            self.headDim = headDim
            overlap = compressRatio == 4
            outputDim = headDim * (overlap ? 2 : 1)
            _wkv.wrappedValue = Linear(config.hiddenSize, outputDim, bias: false)
            _wgate.wrappedValue = Linear(config.hiddenSize, outputDim, bias: false)
            _ape.wrappedValue = zeros([compressRatio, outputDim])
            _norm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
            rope = RoPELayer(
                dims: config.qkRopeHeadDim,
                base: config.compressRopeTheta,
                scalingConfig: config.ropeScaling,
                maxPositionEmbeddings: config.maxPositionEmbeddings,
                freqScale: compressRatio,
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray, poolCache: DeepSeekV4PoolingCache?, offset: Int)
            -> MLXArray
        {
            let batch = x.dim(0)
            let kv = wkv(x)
            let gate = wgate(x)
            let readyKV: MLXArray
            let readyGate: MLXArray
            let poolBase: Int
            if let poolCache {
                (readyKV, readyGate, poolBase) =
                    poolCache.accumulateWindows(kv: kv, gate: gate, offset: offset)
            } else {
                let usable = (kv.dim(1) / compressRatio) * compressRatio
                readyKV = kv[0..., ..<usable, 0...]
                readyGate = gate[0..., ..<usable, 0...]
                poolBase = offset
            }

            let newPooled: MLXArray
            if readyKV.size == 0 {
                newPooled = zeros([batch, 0, headDim], dtype: x.dtype)
            } else {
                let kvWindows = unflatten(readyKV, axis: 1, shape: [-1, compressRatio])
                let gateWindows = unflatten(readyGate, axis: 1, shape: [-1, compressRatio])
                let compressed =
                    overlap
                    ? overlapCompress(kv: kvWindows, gate: gateWindows)
                    : simpleCompress(kv: kvWindows, gate: gateWindows)
                newPooled = rope(norm(compressed)[0..., .newAxis, 0..., 0...], offset: poolBase)
                    .squeezed(axis: 1)
            }
            if let poolCache {
                return poolCache.updateAndFetch(newPooled)
            }
            return newPooled
        }

        private func simpleCompress(kv: MLXArray, gate: MLXArray) -> MLXArray {
            let weights = softmax((gate + ape.asType(gate.dtype)).asType(.float32), axis: -2)
                .asType(kv.dtype)
            return (kv * weights).sum(axis: -2)
        }

        private func overlapCompress(kv: MLXArray, gate: MLXArray) -> MLXArray {
            let batch = kv.dim(0)
            let windows = kv.dim(1)
            let ratio = kv.dim(2)
            let zeroKV = zeros([batch, 1, ratio, headDim], dtype: kv.dtype)
            let kvSplit = split(kv, parts: 2, axis: -1)
            let shiftedKV = concatenated(
                [zeroKV, kvSplit[0][0..., ..<(windows - 1), 0..., 0...]], axis: 1)
            let expandedKV = concatenated([shiftedKV, kvSplit[1]], axis: 2)

            let zeroGate = full(
                [batch, 1, ratio, headDim],
                values: MLXArray(-Float.infinity),
                dtype: gate.dtype,
            )
            let gateWithAPE = gate + ape.asType(gate.dtype)
            let gateSplit = split(gateWithAPE, parts: 2, axis: -1)
            let shiftedGate = concatenated(
                [zeroGate, gateSplit[0][0..., ..<(windows - 1), 0..., 0...]],
                axis: 1,
            )
            let expandedGate = concatenated([shiftedGate, gateSplit[1]], axis: 2)
            let weights = softmax(expandedGate, axis: -2, precise: true).asType(kv.dtype)
            return (expandedKV * weights).sum(axis: -2)
        }
    }

    final class Indexer: Module {
        let config: DeepSeekV4Configuration
        let compressRatio: Int
        let scale: Float

        @ModuleInfo(key: "wq_b") var wqB: Linear
        @ModuleInfo(key: "weights_proj") var weightsProj: Linear
        @ModuleInfo var compressor: Compressor

        init(_ config: DeepSeekV4Configuration, compressRatio: Int) {
            self.config = config
            self.compressRatio = compressRatio
            scale = pow(Float(config.indexHeadDim), -0.5)
            _wqB.wrappedValue = Linear(
                config.qLoraRank,
                config.indexHeads * config.indexHeadDim,
                bias: false,
            )
            _weightsProj.wrappedValue = Linear(config.hiddenSize, config.indexHeads, bias: false)
            _compressor.wrappedValue = Compressor(
                config,
                compressRatio: compressRatio,
                headDim: config.indexHeadDim,
            )
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            qResidual: MLXArray,
            positionRope: RoPELayer,
            poolCache: DeepSeekV4PoolingCache?,
            offset: Int,
        )
            -> MLXArray?
        {
            let batch = x.dim(0)
            let length = x.dim(1)
            let pooled = compressor(x, poolCache: poolCache, offset: offset)
            if pooled.dim(1) == 0 {
                return nil
            }
            var q = wqB(qResidual).reshaped(
                batch,
                length,
                config.indexHeads,
                config.indexHeadDim,
            )
            .transposed(0, 2, 1, 3)
            q = positionRope(q, offset: offset)

            let pooledLength = pooled.dim(1)
            let k = min(config.indexTopK, pooledLength)
            var scores = q.asType(.float32).matmul(
                pooled[0..., .newAxis, 0..., 0...].swappedAxes(-1, -2).asType(.float32),
            )
            scores = maximum(scores, 0) * scale
            let weights = weightsProj(x).asType(.float32) * pow(Float(config.indexHeads), -0.5)
            scores = (scores * weights.swappedAxes(-1, -2)[.ellipsis, .newAxis]).sum(axis: 1)
            if let mask = poolCache?.makePooledMask(length: length, offset: offset) {
                scores = MLX.where(mask[.newAxis, 0..., 0...], scores, MLXArray(-1e30))
            }
            return argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        }
    }

    final class Attention: Module {
        let config: DeepSeekV4Configuration
        let layerIndex: Int
        let compressRatio: Int
        let scale: Float

        @ModuleInfo(key: "wq_a") var wqA: Linear
        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "wq_b") var wqB: Linear
        @ModuleInfo var wkv: Linear
        @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm
        @ModuleInfo(key: "wo_a") var woA: MultiLinear
        @ModuleInfo(key: "wo_b") var woB: Linear
        @ModuleInfo(key: "attn_sink") var attentionSink: MLXArray
        @ModuleInfo var compressor: Compressor?
        @ModuleInfo var indexer: Indexer?
        let rope: RoPELayer

        init(_ config: DeepSeekV4Configuration, layerIndex: Int) {
            self.config = config
            self.layerIndex = layerIndex
            compressRatio = config.compressRatios[layerIndex]
            scale = pow(Float(config.headDim), -0.5)
            _wqA.wrappedValue = Linear(config.hiddenSize, config.qLoraRank, bias: false)
            _qNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: config.rmsNormEps)
            _wqB.wrappedValue = Linear(
                config.qLoraRank,
                config.attentionHeads * config.headDim,
                bias: false,
            )
            _wkv.wrappedValue = Linear(config.hiddenSize, config.headDim, bias: false)
            _kvNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
            _woA.wrappedValue = MultiLinear(
                inputDims: config.attentionHeads * config.headDim / config.outputGroups,
                outputDims: config.outputLoraRank,
                groups: config.outputGroups,
            )
            _woB.wrappedValue = Linear(
                config.outputGroups * config.outputLoraRank,
                config.hiddenSize,
                bias: config.attentionBias,
            )
            _attentionSink.wrappedValue = zeros([config.attentionHeads])
            rope = RoPELayer(
                dims: config.qkRopeHeadDim,
                base: compressRatio == 0 ? config.ropeTheta : config.compressRopeTheta,
                scalingConfig: compressRatio == 0 ? nil : config.ropeScaling,
                maxPositionEmbeddings: config.maxPositionEmbeddings,
            )
            if compressRatio != 0 {
                _compressor.wrappedValue = Compressor(
                    config,
                    compressRatio: compressRatio,
                    headDim: config.headDim,
                )
            }
            if compressRatio != 0, compressRatio != 128 {
                _indexer.wrappedValue = Indexer(config, compressRatio: compressRatio)
            }
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            let batch = x.dim(0)
            let length = x.dim(1)
            let localCache: KVCache?
            let poolCache: DeepSeekV4PoolingCache?
            let indexCache: DeepSeekV4PoolingCache?
            if let list = cache as? CacheList {
                localCache = list[0]
                poolCache = list[1] as? DeepSeekV4PoolingCache
                indexCache =
                    compressRatio != 0 && compressRatio != 128
                    ? list[2] as? DeepSeekV4PoolingCache : nil
            } else {
                localCache = cache
                poolCache = nil
                indexCache = nil
            }
            let offset = localCache?.offset ?? 0

            let qResidual = qNorm(wqA(x))
            var q = wqB(qResidual).reshaped(batch, length, config.attentionHeads, config.headDim)
            q = MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: config.rmsNormEps)
                .transposed(0, 2, 1, 3)
            q = rope(q, offset: offset)

            var kv = kvNorm(wkv(x)).reshaped(batch, 1, length, config.headDim)
            kv = rope(kv, offset: offset)
            if let localCache {
                (kv, _) = localCache.update(
                    keys: kv,
                    values: zeros([batch, 1, length, 0], dtype: kv.dtype),
                )
            }

            let localMask = alignLocalMask(mask.mask, localLength: kv.dim(2))
            let sinks = attentionSink.asType(q.dtype)
            let output: MLXArray
            if compressRatio == 0 {
                output = MLXFast.scaledDotProductAttention(
                    queries: q,
                    keys: kv,
                    values: kv,
                    scale: scale,
                    mask: localMask.map { .array($0) } ?? .none,
                    sinks: sinks,
                )
            } else {
                let pooled = compressor!(x, poolCache: poolCache, offset: offset)
                let pooledMask = poolCache?.makePooledMask(length: length, offset: offset)
                if pooled.dim(1) == 0 {
                    output = MLXFast.scaledDotProductAttention(
                        queries: q,
                        keys: kv,
                        values: kv,
                        scale: scale,
                        mask: localMask.map { .array($0) } ?? .none,
                        sinks: sinks,
                    )
                } else if indexer == nil || pooled.dim(1) <= config.indexTopK {
                    let fullKV = concatenated([kv, pooled[0..., .newAxis, 0..., 0...]], axis: 2)
                    let fullMask = extendMask(
                        localMask, pooledMask: pooledMask, totalLength: fullKV.dim(2))
                    output = MLXFast.scaledDotProductAttention(
                        queries: q,
                        keys: fullKV,
                        values: fullKV,
                        scale: scale,
                        mask: fullMask.map { .array($0) } ?? .none,
                        sinks: sinks,
                    )
                } else {
                    let topK = indexer!(
                        x,
                        qResidual: qResidual,
                        positionRope: rope,
                        poolCache: indexCache,
                        offset: offset,
                    )!
                    let sparseMask = pooledMask.map {
                        takeAlong($0[.newAxis, 0..., 0...], topK, axis: 2)[
                            0...,
                            .newAxis,
                            0...,
                            0...,
                        ]
                    }
                    output = Self.sparsePooledAttention(
                        q: q,
                        localKV: kv,
                        pooled: pooled,
                        topK: topK,
                        localMask: localMask,
                        pooledMask: sparseMask,
                        scale: scale,
                        sinks: sinks,
                    )
                }
            }

            var out = rope(output, offset: offset, inverse: true)
            out = out.reshaped(batch, config.outputGroups, -1, length, config.headDim)
                .transposed(0, 1, 3, 2, 4)
                .flattened(start: -2, end: -1)
            out = woA(out)
            out = out.transposed(0, 2, 1, 3).flattened(start: -2, end: -1)
            return woB(out)
        }

        private func alignLocalMask(_ mask: MLXArray?, localLength: Int) -> MLXArray? {
            guard let mask else {
                return nil
            }
            let current = mask.dim(-1)
            if current == localLength {
                return mask
            }
            if current > localLength {
                return mask[.ellipsis, (current - localLength)...]
            }
            let pad = ones(mask.shape.dropLast() + [localLength - current], dtype: .bool)
            return concatenated([pad, mask], axis: -1)
        }

        private func extendMask(
            _ mask: MLXArray?,
            pooledMask: MLXArray?,
            totalLength: Int,
        )
            -> MLXArray?
        {
            guard let mask else {
                return nil
            }
            let extra = totalLength - mask.dim(-1)
            guard extra > 0 else {
                return mask
            }
            let pool: MLXArray =
                if let pooledMask {
                    pooledMask
                } else {
                    ones([mask.dim(-2), extra], dtype: .bool)
                }
            return concatenated([mask, pool], axis: -1)
        }

        static func sparsePooledAttention(
            q: MLXArray,
            localKV: MLXArray,
            pooled: MLXArray,
            topK: MLXArray,
            localMask: MLXArray?,
            pooledMask: MLXArray?,
            scale: Float,
            sinks: MLXArray,
        )
            -> MLXArray
        {
            let batch = q.dim(0)
            let length = q.dim(2)
            let dim = q.dim(3)
            let idx = topK[0..., .newAxis, 0..., 0..., .newAxis]
            let expandedPooled = broadcast(
                pooled[0..., .newAxis, .newAxis, 0..., 0...],
                to: [batch, 1, length, pooled.dim(1), dim],
            )
            let gathered = takeAlong(
                expandedPooled,
                broadcast(idx, to: idx.shape.dropLast() + [dim]),
                axis: 3,
            )
            let qScaled = q * scale
            var localScores = qScaled.matmul(localKV.swappedAxes(-1, -2))
            localScores = applyScoreMask(localScores, localMask)
            var normalizer = logSumExp(localScores, axis: -1, keepDims: true)
            let pooledScores = qScaled.transposed(0, 2, 1, 3)
                .matmul(gathered.squeezed(axis: 1).swappedAxes(-1, -2))
                .transposed(0, 2, 1, 3)
            let maskedPooledScores = applyScoreMask(pooledScores, pooledMask)
            normalizer = logAddExp(
                normalizer,
                logSumExp(maskedPooledScores, axis: -1, keepDims: true),
            )
            normalizer = logAddExp(normalizer, sinks[.newAxis, 0..., .newAxis, .newAxis])
            let localWeights = exp(localScores - normalizer)
            let pooledWeights = exp(maskedPooledScores - normalizer)
            var out = localWeights.matmul(localKV)
            out =
                out
                + pooledWeights.transposed(0, 2, 1, 3)
                .matmul(gathered.squeezed(axis: 1))
                .transposed(0, 2, 1, 3)
            return out.asType(q.dtype)
        }

        private static func applyScoreMask(_ scores: MLXArray, _ mask: MLXArray?) -> MLXArray {
            guard let mask else {
                return scores
            }
            return MLX.where(mask, scores, MLXArray(-1e30))
        }
    }

    final class Block: Module {
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo(key: "ffn") var feedForward: MoE
        @ModuleInfo(key: "attn_norm") var attentionNorm: RMSNorm
        @ModuleInfo(key: "ffn_norm") var feedForwardNorm: RMSNorm
        @ModuleInfo(key: "attn_hc") var attentionHC: HyperConnection
        @ModuleInfo(key: "ffn_hc") var feedForwardHC: HyperConnection

        init(_ config: DeepSeekV4Configuration, layerIndex: Int) {
            _attention.wrappedValue = Attention(config, layerIndex: layerIndex)
            _feedForward.wrappedValue = MoE(config, layerIndex: layerIndex)
            _attentionNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            _feedForwardNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            _attentionHC.wrappedValue = HyperConnection(config)
            _feedForwardHC.wrappedValue = HyperConnection(config)
            super.init()
        }

        func callAsFunction(
            _ hidden: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
            inputIDs: MLXArray,
        )
            -> MLXArray
        {
            var residual = hidden
            var (x, post, comb) = attentionHC(hidden)
            x = attention(attentionNorm(x), mask: mask, cache: cache)
            var h = hcExpand(x, residual: residual, post: post, comb: comb)

            residual = h
            (x, post, comb) = feedForwardHC(h)
            x = feedForward(feedForwardNorm(x), inputIDs: inputIDs)
            h = hcExpand(x, residual: residual, post: post, comb: comb)
            return h
        }
    }

    final class Model: Module {
        let config: DeepSeekV4Configuration

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo var layers: [Block]
        @ModuleInfo var norm: RMSNorm
        @ModuleInfo(key: "hc_head") var hcHead: HyperHead

        init(_ config: DeepSeekV4Configuration) {
            self.config = config
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
                Block(config, layerIndex: $0)
            }
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _hcHead.wrappedValue = HyperHead(config)
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray,
            cache: [KVCache]? = nil,
            inputEmbeddings: MLXArray? = nil,
        )
            -> MLXArray
        {
            var hidden = inputEmbeddings ?? embedTokens(inputs)
            hidden = broadcast(
                hidden[0..., 0..., .newAxis, 0...],
                to: [hidden.dim(0), hidden.dim(1), config.hcMult, hidden.dim(2)],
            )
            let firstCache: KVCache? = {
                guard let cache = cache?.first else {
                    return nil
                }
                if let list = cache as? CacheList {
                    return list[0]
                }
                return cache
            }()
            let mask = makeAttentionMask(
                n: hidden.dim(1),
                cache: firstCache,
                windowSize: config.slidingWindow,
                returnArray: true,
            )
            for (index, layer) in layers.enumerated() {
                hidden = layer(hidden, mask: mask, cache: cache?[index], inputIDs: inputs)
            }
            return norm(hcHead(hidden))
        }
    }

    final class LanguageModel: Module {
        let config: DeepSeekV4Configuration

        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear

        init(_ config: DeepSeekV4Configuration) {
            self.config = config
            _model.wrappedValue = Model(config)
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
            super.init()
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
            lmHead(model(inputs, cache: cache))
        }
    }
}

public final class DeepSeekV4: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let config: DeepSeekV4Configuration

    @ModuleInfo(key: "language_model") private var languageModel: DeepSeekV4Language.LanguageModel

    public init(_ config: DeepSeekV4Configuration) {
        self.config = config
        vocabularySize = config.vocabularySize
        kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        _languageModel.wrappedValue = DeepSeekV4Language.LanguageModel(config)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(configuration: KVCacheConfiguration?) -> [KVCache] {
        (0 ..< config.hiddenLayers).map { layerIndex in
            let ratio = config.compressRatios[layerIndex]
            let local = RotatingKVCache(maxSize: config.slidingWindow)
            if ratio == 0 {
                return local
            }
            if ratio == 128 {
                return CacheList(local, DeepSeekV4PoolingCache(ratio: ratio))
            }
            return CacheList(
                local,
                DeepSeekV4PoolingCache(ratio: ratio),
                DeepSeekV4PoolingCache(ratio: ratio),
            )
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var filtered: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("mtp.") || key.contains("rotary_emb.inv_freq") {
                continue
            }
            let parts = key.split(separator: ".")
            if parts.count >= 2, parts[0] == "layers", let layer = Int(parts[1]),
                layer >= config.hiddenLayers
            {
                continue
            }
            filtered[key] = value
        }

        var remapped: [String: MLXArray] = [:]
        let topRemap = [
            "embed.weight": "model.embed_tokens.weight",
            "norm.weight": "model.norm.weight",
            "head.weight": "lm_head.weight",
            "hc_head_fn": "model.hc_head.fn",
            "hc_head_base": "model.hc_head.base",
            "hc_head_scale": "model.hc_head.scale",
        ]

        for (key, value) in filtered {
            if key.contains("tid2eid") {
                remapped[key] = value.asType(.int32)
                continue
            }
            if key.hasSuffix(".scale") {
                let weightKey = String(key.dropLast(".scale".count)) + ".weight"
                if let weight = filtered[weightKey], weight.dtype == .uint8 || weight.dtype == .int8
                {
                    remapped[key + "s"] = value
                    remapped[weightKey] =
                        weight.dtype == .uint8 ? weight.view(dtype: .uint32) : weight
                    continue
                }
            }
            remapped[key] = value
        }

        for (old, new) in topRemap where remapped[old] != nil {
            remapped[new] = remapped.removeValue(forKey: old)
        }

        var normalized: [String: MLXArray] = [:]
        for (key, value) in remapped {
            var newKey = key.hasPrefix("layers.") ? "model.\(key)" : key
            if newKey.hasSuffix(".ffn.gate.bias") {
                newKey =
                    String(newKey.dropLast(".ffn.gate.bias".count))
                    + ".ffn.gate.e_score_correction_bias"
            }
            for sub in ["attn", "ffn"] {
                for param in ["fn", "base", "scale"] {
                    newKey = newKey.replacingOccurrences(
                        of: ".hc_\(sub)_\(param)",
                        with: ".\(sub)_hc.\(param)",
                    )
                }
            }
            for (old, new) in ["w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"] {
                newKey = newKey.replacingOccurrences(
                    of: ".shared_experts.\(old).",
                    with: ".shared_experts.\(new).",
                )
            }
            if newKey.hasPrefix("model.") || newKey.hasPrefix("lm_head.") {
                normalized["language_model.\(newKey)"] = value
            } else if newKey.hasPrefix("language_model.") {
                normalized[newKey] = value
            } else {
                normalized[newKey] = value
            }
        }

        stackLegacyExperts(weights: &normalized)
        reshapeOutputLoRA(weights: &normalized)
        return normalized
    }

    private func stackLegacyExperts(weights: inout [String: MLXArray]) {
        for layerIndex in 0 ..< config.hiddenLayers {
            let prefix = "language_model.model.layers.\(layerIndex).ffn.experts"
            for (source, destination) in [
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ] {
                for suffix in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).0.\(source).\(suffix)"
                    guard weights[firstKey] != nil else {
                        continue
                    }
                    let tensors = (0 ..< config.nRoutedExperts).map { expert in
                        weights.removeValue(forKey: "\(prefix).\(expert).\(source).\(suffix)")!
                    }
                    weights[
                        "language_model.model.layers.\(layerIndex).ffn.switch_mlp.\(destination).\(suffix)",
                    ] = MLX.stacked(tensors)
                }
            }
        }
    }

    private func reshapeOutputLoRA(weights: inout [String: MLXArray]) {
        for layerIndex in 0 ..< config.hiddenLayers {
            let prefix = "language_model.model.layers.\(layerIndex).attn.wo_a"
            for suffix in ["weight", "scales", "biases"] {
                let key = "\(prefix).\(suffix)"
                if let value = weights[key], value.ndim == 2 {
                    weights[key] = value.reshaped(config.outputGroups, config.outputLoraRank, -1)
                }
            }
        }
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

extension StringOrNumber {
    fileprivate func asString() -> String? {
        if case .string(let value) = self {
            return value
        }
        return nil
    }
}
