import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Nemotron-H Language Wrapper

private enum NemotronOmniHBlockType {
    case mamba
    case attention
    case mlp
    case moe

    init(from char: Character) {
        switch char {
        case "M": self = .mamba
        case "*": self = .attention
        case "-": self = .mlp
        case "E": self = .moe
        default: preconditionFailure("Unknown Nemotron-H block type: \(char)")
        }
    }
}

private protocol NemotronOmniHMixer: Module {
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?,
    )
        -> MLXArray
}

private func nemotronOmniReluSquared(_ x: MLXArray) -> MLXArray {
    let y = MLX.maximum(x, MLXArray(0))
    return y * y
}

private final class NemotronOmniHRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let groupSize: Int

    init(dimensions: Int, eps: Float, groupSize: Int) {
        self.eps = eps
        self.groupSize = groupSize
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray?) -> MLXArray {
        var states = x
        if let gate {
            states = states * silu(gate)
        }

        let shape = states.shape
        var groupedShape = Array(shape.dropLast())
        groupedShape.append(-1)
        groupedShape.append(groupSize)
        let grouped = states.reshaped(groupedShape)
        let normalized = MLXFast.rmsNorm(
            grouped,
            weight: MLXArray.ones([groupSize], dtype: grouped.dtype),
            eps: eps,
        )
        return weight * normalized.reshaped(shape)
    }
}

private final class NemotronOmniHMamba2Mixer: Module, NemotronOmniHMixer {
    let numHeads: Int
    let hiddenSize: Int
    let ssmStateSize: Int
    let convKernelSize: Int
    let intermediateSize: Int
    let numGroups: Int
    let headDim: Int
    let timeStepLimit: (Float, Float)
    let headsPerGroup: Int
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "D") var d: MLXArray
    @ModuleInfo(key: "norm") var norm: NemotronOmniHRMSNormGated

    init(_ args: NemotronHConfiguration) {
        numHeads = args.mambaNumHeads
        hiddenSize = args.hiddenSize
        ssmStateSize = args.ssmStateSize
        convKernelSize = args.convKernel
        intermediateSize = args.mambaNumHeads * args.mambaHeadDim
        numGroups = args.nGroups
        headDim = args.mambaHeadDim
        timeStepLimit = (args.timeStepLimitMin, args.timeStepLimitMax)
        headsPerGroup = numHeads / numGroups
        convDim = intermediateSize + 2 * numGroups * ssmStateSize

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            groups: convDim,
            bias: args.useConvBias,
        )
        _inProj.wrappedValue = Linear(
            hiddenSize,
            intermediateSize + convDim + numHeads,
            bias: args.mambaProjBias,
        )
        _outProj.wrappedValue = Linear(
            intermediateSize,
            hiddenSize,
            bias: args.mambaProjBias,
        )
        _dtBias.wrappedValue = MLXArray.ones([numHeads])
        _aLog.wrappedValue = MLX.log(MLXArray(0 ..< numHeads).asType(.float32) + 1)
        _d.wrappedValue = MLXArray.ones([numHeads])
        _norm.wrappedValue = NemotronOmniHRMSNormGated(
            dimensions: intermediateSize,
            eps: args.layerNormEpsilon,
            groupSize: intermediateSize / numGroups,
        )
        super.init()
    }

    private func applyConv(_ input: MLXArray, mask: MLXArray?, cache: MambaCache?) -> MLXArray {
        var convInput = input
        if let mask {
            convInput = MLX.where(
                expandedDimensions(mask, axis: -1),
                convInput,
                MLXArray.zeros(like: convInput),
            )
        }

        let batch = convInput.dim(0)
        let dtype = convInput.dtype
        var convState = cache?[0]
        if convState == nil {
            convState = MLXArray.zeros(
                [batch, max(convKernelSize - 1, 0), convDim],
                dtype: dtype,
            )
        }

        let paddedInput = concatenated([convState!, convInput], axis: 1)
        if let cache {
            let end = paddedInput.dim(1)
            let start = max(0, end - max(convKernelSize - 1, 0))
            cache[0] = contiguous(paddedInput[0..., start ..< end, 0...])
        }
        return silu(conv1d(paddedInput))
    }

    private func mambaForward(_ hiddenStates: MLXArray, mask: MLXArray?, cache: MambaCache?)
        -> MLXArray
    {
        let projected = inProj(hiddenStates)
        let splits = split(
            projected,
            indices: [intermediateSize, intermediateSize + convDim],
            axis: -1,
        )
        let gate = splits[0]
        let convInput = splits[1]
        let dt = splits[2]

        let convOutput = applyConv(convInput, mask: mask, cache: cache)
        let convSplits = split(
            convOutput,
            indices: [intermediateSize, intermediateSize + numGroups * ssmStateSize],
            axis: -1,
        )

        var hidden = convSplits[0]
        var b = convSplits[1]
        var c = convSplits[2]
        hidden = hidden.reshaped(hidden.dim(0), hidden.dim(1), numHeads, headDim)
        b = b.reshaped(b.dim(0), b.dim(1), numGroups, ssmStateSize)
        c = c.reshaped(c.dim(0), c.dim(1), numGroups, ssmStateSize)

        let (y, nextState) = ssmUpdate(
            hiddenStates: hidden,
            ALog: aLog,
            B: b,
            C: c,
            D: d,
            dt: dt.reshaped(dt.dim(0), dt.dim(1), numHeads),
            dtBias: dtBias,
            state: cache?[1],
            timeStepLimit: timeStepLimit,
            mask: mask,
        )

        if let cache {
            cache[1] = nextState
            cache.advance(hiddenStates.dim(1))
        }

        return outProj(norm(y.flattened(start: 2), gate: gate))
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask _: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?,
    )
        -> MLXArray
    {
        mambaForward(x, mask: ssmMask, cache: cache as? MambaCache)
    }
}

private final class NemotronOmniHAttention: Module, NemotronOmniHMixer {
    let scale: Float
    let numHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    init(_ args: NemotronHConfiguration) {
        numHeads = args.numAttentionHeads
        numKeyValueHeads = args.numKeyValueHeads
        headDim = args.headDim ?? (args.hiddenSize / args.numAttentionHeads)
        scale = pow(Float(headDim), -0.5)
        _wq.wrappedValue = Linear(
            args.hiddenSize,
            numHeads * headDim,
            bias: args.attentionBias,
        )
        _wk.wrappedValue = Linear(
            args.hiddenSize,
            numKeyValueHeads * headDim,
            bias: args.attentionBias,
        )
        _wv.wrappedValue = Linear(
            args.hiddenSize,
            numKeyValueHeads * headDim,
            bias: args.attentionBias,
        )
        _wo.wrappedValue = Linear(
            numHeads * headDim,
            args.hiddenSize,
            bias: args.attentionBias,
        )
        super.init()
    }

    private func attentionForward(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
    )
        -> MLXArray
    {
        let batch = x.dim(0)
        let length = x.dim(1)
        let queries = wq(x).reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        let keys = wk(x).reshaped(batch, length, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(batch, length, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)
        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask,
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, -1)
        return wo(output)
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask _: MLXArray?,
        cache: KVCache?,
    )
        -> MLXArray
    {
        attentionForward(x, mask: attentionMask, cache: cache)
    }
}

private final class NemotronOmniHMLP: Module, UnaryLayer, NemotronOmniHMixer {
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ args: NemotronHConfiguration, intermediateSize: Int? = nil) {
        let intermediate = intermediateSize ?? args.intermediateSize
        _upProj.wrappedValue = Linear(args.hiddenSize, intermediate, bias: args.mlpBias)
        _downProj.wrappedValue = Linear(intermediate, args.hiddenSize, bias: args.mlpBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(nemotronOmniReluSquared(upProj(x)))
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask _: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask _: MLXArray?,
        cache _: KVCache?,
    )
        -> MLXArray
    {
        callAsFunction(x)
    }
}

private func nemotronOmniGroupExpertSelect(
    gates: MLXArray,
    correctionBias: MLXArray,
    topK: Int,
    groups: Int,
    topKGroups: Int,
    routedScalingFactor: Float,
    normalizeTopKProbability: Bool,
)
    -> (MLXArray, MLXArray)
{
    let batch = gates.dim(0)
    let sequenceLength = gates.dim(1)
    let originalScores = sigmoid(gates.asType(.float32))
    var scores = originalScores + correctionBias

    if groups > 1 {
        let expertsPerGroup = scores.dim(-1) / groups
        let groupedScores = scores.reshaped(batch, sequenceLength, groups, -1)
        let topGroupedScores = sorted(groupedScores, axis: -1)[.ellipsis, ..<2]
            .sum(axis: -1, keepDims: true)
        let discardedGroupCount = groups - topKGroups
        var groupIndices = argPartition(
            topGroupedScores,
            kth: discardedGroupCount - 1,
            axis: -2,
        )[.ellipsis, ..<discardedGroupCount, 0...]
        groupIndices = broadcast(
            groupIndices,
            to: [batch, sequenceLength, discardedGroupCount, expertsPerGroup],
        )
        scores = putAlong(groupedScores, groupIndices, values: MLXArray(0.0), axis: -2)
        scores = flattened(scores, start: -2, end: -1)
    }

    let indices = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
    var selectedScores = takeAlong(originalScores, indices, axis: -1)
    if topK > 1, normalizeTopKProbability {
        selectedScores = selectedScores / (selectedScores.sum(axis: -1, keepDims: true) + 1e-20)
    }
    return (indices, selectedScores * routedScalingFactor)
}

private final class NemotronOmniHMoEGate: Module {
    let topK: Int
    let groups: Int
    let topKGroups: Int
    let routedScalingFactor: Float
    let normalizeTopKProbability: Bool

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray

    init(_ args: NemotronHConfiguration) {
        topK = args.numExpertsPerTok
        groups = args.nGroup
        topKGroups = args.topkGroup
        routedScalingFactor = args.routedScalingFactor
        normalizeTopKProbability = args.normTopkProb
        _weight.wrappedValue = MLXArray.zeros([args.nRoutedExperts, args.hiddenSize])
        _correctionBias.wrappedValue = MLXArray.zeros([args.nRoutedExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        nemotronOmniGroupExpertSelect(
            gates: MLX.matmul(x, weight.transposed()),
            correctionBias: correctionBias,
            topK: topK,
            groups: groups,
            topKGroups: topKGroups,
            routedScalingFactor: routedScalingFactor,
            normalizeTopKProbability: normalizeTopKProbability,
        )
    }
}

private final class NemotronOmniHSwitchMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: SwitchLinear
    @ModuleInfo(key: "fc2") var fc2: SwitchLinear

    init(inputDims: Int, hiddenDims: Int, numExperts: Int) {
        _fc1.wrappedValue = SwitchLinear(
            inputDims: inputDims,
            outputDims: hiddenDims,
            numExperts: numExperts,
            bias: false,
        )
        _fc2.wrappedValue = SwitchLinear(
            inputDims: hiddenDims,
            outputDims: inputDims,
            numExperts: numExperts,
            bias: false,
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])
        let shouldSort = indices.size > 64
        var sorted = indices
        var inverseOrder = MLXArray()

        if shouldSort {
            (x, sorted, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        var y = fc1(x, sorted, sortedIndices: shouldSort)
        y = nemotronOmniReluSquared(y)
        y = fc2(y, sorted, sortedIndices: shouldSort)
        if shouldSort {
            y = scatterUnsort(x: y, invOrder: inverseOrder, shape: indices.shape)
        }
        return MLX.squeezed(y, axis: -2)
    }
}

private final class NemotronOmniHMoE: Module, UnaryLayer, NemotronOmniHMixer {
    @ModuleInfo(key: "gate") var gate: NemotronOmniHMoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: NemotronOmniHSwitchMLP
    @ModuleInfo(key: "shared_experts") var sharedExperts: NemotronOmniHMLP?
    @ModuleInfo(key: "fc1_latent_proj") var latentInput: Linear?
    @ModuleInfo(key: "fc2_latent_proj") var latentOutput: Linear?

    init(_ args: NemotronHConfiguration) {
        _gate.wrappedValue = NemotronOmniHMoEGate(args)
        _switchMLP.wrappedValue = NemotronOmniHSwitchMLP(
            inputDims: args.moeLatentSize ?? args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.nRoutedExperts,
        )
        if let latentSize = args.moeLatentSize {
            _latentInput.wrappedValue = Linear(args.hiddenSize, latentSize, bias: args.mlpBias)
            _latentOutput.wrappedValue = Linear(latentSize, args.hiddenSize, bias: args.mlpBias)
        }
        if let shared = args.nSharedExperts, shared > 0 {
            _sharedExperts.wrappedValue = NemotronOmniHMLP(
                args,
                intermediateSize: args.moeSharedExpertIntermediateSize,
            )
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var y = switchMLP(latentInput?(x) ?? x, indices)
        y = weightedExpertSum(y, scores).asType(y.dtype)
        if let latentOutput { y = latentOutput(y) }
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask _: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask _: MLXArray?,
        cache _: KVCache?,
    )
        -> MLXArray
    {
        callAsFunction(x)
    }
}

private final class NemotronOmniHBlock: Module {
    let blockType: NemotronOmniHBlockType

    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "mixer") var mixer: Module

    init(_ args: NemotronHConfiguration, blockType: Character) {
        self.blockType = NemotronOmniHBlockType(from: blockType)
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        switch self.blockType {
        case .mamba:
            _mixer.wrappedValue = NemotronOmniHMamba2Mixer(args)
        case .attention:
            _mixer.wrappedValue = NemotronOmniHAttention(args)
        case .mlp:
            _mixer.wrappedValue = NemotronOmniHMLP(args)
        case .moe:
            _mixer.wrappedValue = NemotronOmniHMoE(args)
        }
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?,
    )
        -> MLXArray
    {
        let mixer = mixer as! NemotronOmniHMixer
        return x + mixer(norm(x), attentionMask: attentionMask, ssmMask: ssmMask, cache: cache)
    }
}

private final class NemotronOmniHBackbone: Module {
    let args: NemotronHConfiguration
    let firstAttentionCacheIndex: Int?
    let firstMambaCacheIndex: Int?

    @ModuleInfo(key: "embeddings") var embeddings: Embedding
    @ModuleInfo(key: "layers") var layers: [NemotronOmniHBlock]
    @ModuleInfo(key: "norm_f") var normF: RMSNorm

    init(_ args: NemotronHConfiguration) {
        self.args = args
        _embeddings.wrappedValue = Embedding(
            embeddingCount: args.vocabSize,
            dimensions: args.hiddenSize,
        )
        let pattern = Array(args.hybridOverridePattern)
        _layers.wrappedValue = pattern.enumerated().map { index, type in
            NemotronOmniHBlock(args.configuration(forLayer: index), blockType: type)
        }
        _normF.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)

        var attentionIndex: Int? = nil
        var mambaCount = 0
        for char in pattern {
            if char == "*" {
                attentionIndex = mambaCount
                break
            } else if char == "M" {
                mambaCount += 1
            }
        }
        firstAttentionCacheIndex = attentionIndex

        var mambaIndex: Int? = nil
        var attentionCount = 0
        for char in pattern {
            if char == "M" {
                mambaIndex = attentionCount
                break
            } else if char == "*" {
                attentionCount += 1
            }
        }
        firstMambaCacheIndex = mambaIndex
        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray?,
        cache: [KVCache]? = nil,
        inputEmbeddings: MLXArray? = nil,
    )
        throws -> MLXArray
    {
        let hidden: MLXArray
        if let inputEmbeddings {
            hidden = inputEmbeddings
        } else if let inputs {
            hidden = embeddings(inputs)
        } else {
            throw VLMError.processing("Nemotron Omni requires input IDs or input embeddings.")
        }
        var currentHidden = hidden

        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode = {
            guard let cacheIndex = firstAttentionCacheIndex,
                let cache,
                cacheIndex < cache.count
            else {
                return .none
            }
            return createAttentionMask(h: currentHidden, cache: cache[cacheIndex])
        }()

        let ssmMask: MLXArray? = {
            guard let cacheIndex = firstMambaCacheIndex,
                let cache,
                cacheIndex < cache.count,
                let mambaCache = cache[cacheIndex] as? MambaCache
            else {
                return nil
            }
            return mambaCache.makeMask(N: currentHidden.dim(1))
        }()

        var cacheCounter = 0
        for layer in layers {
            let layerCache: KVCache?
            switch layer.blockType {
            case .mamba, .attention:
                layerCache = cache?[cacheCounter]
                cacheCounter += 1
            case .mlp, .moe:
                layerCache = nil
            }
            currentHidden = layer(
                currentHidden,
                attentionMask: attentionMask,
                ssmMask: ssmMask,
                cache: layerCache,
            )
        }

        return normF(currentHidden)
    }
}

private final class NemotronOmniLanguageModel: Module, KVCacheDimensionProvider {
    let configuration: NemotronHConfiguration
    let vocabularySize: Int
    let kvHeads: [Int]

    @ModuleInfo(key: "backbone") var backbone: NemotronOmniHBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    var loraLayers: [Module] {
        backbone.layers
    }

    init(_ args: NemotronHConfiguration) {
        configuration = args
        vocabularySize = args.vocabSize
        kvHeads = Array(args.hybridOverridePattern).compactMap { char in
            let blockType = NemotronOmniHBlockType(from: char)
            switch blockType {
            case .attention:
                return args.numKeyValueHeads
            case .mamba:
                return 0
            case .mlp, .moe:
                return nil
            }
        }
        _backbone.wrappedValue = NemotronOmniHBackbone(args)
        _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
        super.init()
    }

    func inputEmbeddings(_ inputIDs: MLXArray) -> MLXArray {
        backbone.embeddings(inputIDs)
    }

    func callAsFunction(
        _ inputs: MLXArray?,
        cache: [KVCache]? = nil,
        inputEmbeddings: MLXArray? = nil,
    )
        throws -> MLXArray
    {
        try lmHead(backbone(inputs, cache: cache, inputEmbeddings: inputEmbeddings))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        do {
            return try callAsFunction(inputs, cache: cache, inputEmbeddings: nil)
        } catch {
            preconditionFailure(error.localizedDescription)
        }
    }

    func newCache(configuration _: KVCacheConfiguration?) -> [KVCache] {
        Array(configuration.hybridOverridePattern).compactMap { char in
            switch NemotronOmniHBlockType(from: char) {
            case .mamba:
                MambaCache()
            case .attention:
                KVCacheSimple()
            case .mlp, .moe:
                nil
            }
        }
    }

    func sanitizeLanguageWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        for (key, value) in weights where key.contains("conv1d.weight") && value.dim(-1) != 1 {
            sanitized[key] = value.swappedAxes(1, 2)
        }

        for layer in 0 ..< configuration.numHiddenLayers {
            let prefix = "backbone.layers.\(layer).mixer"
            let count = configuration.configuration(forLayer: layer).nRoutedExperts
            for (source, target) in [("down_proj", "fc2"), ("up_proj", "fc1")] {
                for name in [
                    "weight", "scales", "biases", "bias", "weight_global_scale", "weight_scale_2",
                ] {
                    let keys = (0 ..< count).map { "\(prefix).experts.\($0).\(source).\(name)" }
                    // Keep incomplete source sets intact for the strict loader to reject.
                    guard !keys.isEmpty, keys.allSatisfy({ sanitized[$0] != nil }) else { continue }
                    sanitized["\(prefix).switch_mlp.\(target).\(name)"] = stacked(
                        keys.map { sanitized.removeValue(forKey: $0)! })
                }
            }
        }
        sanitized = sanitized.filter { !$0.key.hasPrefix("mtp.") }

        return sanitized
    }

    var castPredicate: ((String) -> Bool)? {
        { key in
            !key.contains("e_score_correction_bias") && !key.contains("A_log")
        }
    }
}

// MARK: - Configuration

public struct NemotronOmniConfiguration: Decodable, Sendable {
    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let intermediateSize: Int
        public let imageSize: Int
        public let patchSize: Int
        public let maxResolution: Int
        public let downsampleRatio: Float
        public let dynamicImageMinNumPatches: Int
        public let dynamicImageMaxNumPatches: Int
        public let videoTargetNumPatches: Int
        public let videoTemporalPatchSize: Int
        public let numClassTokens: Int
        public let numRegisters: Int
        public let layerNormEpsilon: Float

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case intermediateSize = "intermediate_size"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case maxResolution = "max_resolution"
            case downsampleRatio = "downsample_ratio"
            case dynamicImageMinNumPatches = "dynamic_image_min_num_patches"
            case dynamicImageMaxNumPatches = "dynamic_image_max_num_patches"
            case minNumPatches = "min_num_patches"
            case maxNumPatches = "max_num_patches"
            case videoTargetNumPatches = "video_target_num_patches"
            case videoTemporalPatchSize = "video_temporal_patch_size"
            case numClassTokens = "num_cls_tokens"
            case numRegisters = "num_registers"
            case layerNormEpsilon = "layer_norm_eps"
        }

        public init(
            modelType: String = "radio",
            hiddenSize: Int = 1280,
            numHiddenLayers: Int = 32,
            numAttentionHeads: Int = 16,
            intermediateSize: Int = 5120,
            imageSize: Int = 224,
            patchSize: Int = 16,
            maxResolution: Int = 2048,
            downsampleRatio: Float = 0.5,
            dynamicImageMinNumPatches: Int = 1024,
            dynamicImageMaxNumPatches: Int = 13312,
            videoTargetNumPatches: Int = 1024,
            videoTemporalPatchSize: Int = 2,
            numClassTokens: Int = 1,
            numRegisters: Int = 0,
            layerNormEpsilon: Float = 1e-6,
        ) {
            self.modelType = modelType
            self.hiddenSize = hiddenSize
            self.numHiddenLayers = numHiddenLayers
            self.numAttentionHeads = numAttentionHeads
            self.intermediateSize = intermediateSize
            self.imageSize = imageSize
            self.patchSize = patchSize
            self.maxResolution = maxResolution
            self.downsampleRatio = downsampleRatio
            self.dynamicImageMinNumPatches = dynamicImageMinNumPatches
            self.dynamicImageMaxNumPatches = dynamicImageMaxNumPatches
            self.videoTargetNumPatches = videoTargetNumPatches
            self.videoTemporalPatchSize = videoTemporalPatchSize
            self.numClassTokens = numClassTokens
            self.numRegisters = numRegisters
            self.layerNormEpsilon = layerNormEpsilon
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "radio"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1280
            numHiddenLayers =
                try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 32
            numAttentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 5120
            imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 224
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
            maxResolution =
                try c.decodeIfPresent(Int.self, forKey: .maxResolution) ?? 2048
            downsampleRatio =
                try c.decodeIfPresent(Float.self, forKey: .downsampleRatio) ?? 0.5
            dynamicImageMinNumPatches =
                try c.decodeIfPresent(Int.self, forKey: .dynamicImageMinNumPatches)
                ?? c.decodeIfPresent(Int.self, forKey: .minNumPatches)
                ?? 1024
            dynamicImageMaxNumPatches =
                try c.decodeIfPresent(Int.self, forKey: .dynamicImageMaxNumPatches)
                ?? c.decodeIfPresent(Int.self, forKey: .maxNumPatches)
                ?? 13312
            videoTargetNumPatches =
                try c.decodeIfPresent(Int.self, forKey: .videoTargetNumPatches) ?? 1024
            videoTemporalPatchSize =
                try c.decodeIfPresent(Int.self, forKey: .videoTemporalPatchSize) ?? 2
            numClassTokens = try c.decodeIfPresent(Int.self, forKey: .numClassTokens) ?? 1
            numRegisters = try c.decodeIfPresent(Int.self, forKey: .numRegisters) ?? 0
            layerNormEpsilon = try c.decodeIfPresent(Float.self, forKey: .layerNormEpsilon) ?? 1e-6
        }

        public var downsampleFactor: Int {
            max(1, Int(round(1 / Double(max(downsampleRatio, 0.0001)))))
        }
    }

    public struct TextConfiguration: Decodable, Sendable {
        public let hiddenSize: Int

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
        }

        public init(hiddenSize: Int = 6144) {
            self.hiddenSize = hiddenSize
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 6144
        }
    }

    public struct SoundConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let numAttentionHeads: Int
        public let numHiddenLayers: Int
        public let intermediateSize: Int
        public let hiddenActivation: String
        public let attentionBias: Bool
        public let convolutionBias: Bool
        public let convKernelSize: Int
        public let subsamplingFactor: Int
        public let subsamplingConvChannels: Int
        public let numMelBins: Int
        public let featureSize: Int
        public let subsamplingConvKernelSize: Int
        public let subsamplingConvStride: Int
        public let maxPositionEmbeddings: Int
        public let scaleInput: Bool
        public let projectionHiddenSize: Int
        public let projectionBias: Bool
        public let samplingRate: Int
        public let hopLength: Int
        public let fftLength: Int
        public let windowLength: Int
        public let preemphasis: Float?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case numAttentionHeads = "num_attention_heads"
            case numHiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case hiddenActivation = "hidden_act"
            case attentionBias = "attention_bias"
            case convolutionBias = "convolution_bias"
            case convKernelSize = "conv_kernel_size"
            case subsamplingFactor = "subsampling_factor"
            case subsamplingConvChannels = "subsampling_conv_channels"
            case numMelBins = "num_mel_bins"
            case featureSize = "feat_in"
            case subsamplingConvKernelSize = "subsampling_conv_kernel_size"
            case subsamplingConvStride = "subsampling_conv_stride"
            case maxPositionEmbeddings = "max_position_embeddings"
            case scaleInput = "scale_input"
            case projectionHiddenSize = "projection_hidden_size"
            case projectionBias = "projection_bias"
            case samplingRate = "sampling_rate"
            case hopLength = "hop_length"
            case fftLength = "n_fft"
            case windowLength = "win_length"
            case preemphasis
        }

        public init(
            modelType: String = "parakeet",
            hiddenSize: Int = 1024,
            numAttentionHeads: Int = 8,
            numHiddenLayers: Int = 24,
            intermediateSize: Int = 4096,
            hiddenActivation: String = "silu",
            attentionBias: Bool = false,
            convolutionBias: Bool = false,
            convKernelSize: Int = 9,
            subsamplingFactor: Int = 8,
            subsamplingConvChannels: Int = 256,
            numMelBins: Int = 128,
            featureSize: Int? = nil,
            subsamplingConvKernelSize: Int = 3,
            subsamplingConvStride: Int = 2,
            maxPositionEmbeddings: Int = 5000,
            scaleInput: Bool = false,
            projectionHiddenSize: Int = 4096,
            projectionBias: Bool = false,
            samplingRate: Int = 16000,
            hopLength: Int = 160,
            fftLength: Int = 512,
            windowLength: Int = 400,
            preemphasis: Float? = 0.97,
        ) {
            self.modelType = modelType
            self.hiddenSize = hiddenSize
            self.numAttentionHeads = numAttentionHeads
            self.numHiddenLayers = numHiddenLayers
            self.intermediateSize = intermediateSize
            self.hiddenActivation = hiddenActivation
            self.attentionBias = attentionBias
            self.convolutionBias = convolutionBias
            self.convKernelSize = convKernelSize
            self.subsamplingFactor = subsamplingFactor
            self.subsamplingConvChannels = subsamplingConvChannels
            self.numMelBins = numMelBins
            self.featureSize = featureSize ?? numMelBins
            self.subsamplingConvKernelSize = subsamplingConvKernelSize
            self.subsamplingConvStride = subsamplingConvStride
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.scaleInput = scaleInput
            self.projectionHiddenSize = projectionHiddenSize
            self.projectionBias = projectionBias
            self.samplingRate = samplingRate
            self.hopLength = hopLength
            self.fftLength = fftLength
            self.windowLength = windowLength
            self.preemphasis = preemphasis
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "parakeet"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
            numAttentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 8
            numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 24
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4096
            hiddenActivation =
                try c.decodeIfPresent(String.self, forKey: .hiddenActivation) ?? "silu"
            attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            convolutionBias =
                try c.decodeIfPresent(Bool.self, forKey: .convolutionBias) ?? false
            convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 9
            subsamplingFactor =
                try c.decodeIfPresent(Int.self, forKey: .subsamplingFactor) ?? 8
            subsamplingConvChannels =
                try c.decodeIfPresent(Int.self, forKey: .subsamplingConvChannels) ?? 256
            numMelBins = try c.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
            featureSize = try c.decodeIfPresent(Int.self, forKey: .featureSize) ?? numMelBins
            subsamplingConvKernelSize =
                try c.decodeIfPresent(Int.self, forKey: .subsamplingConvKernelSize) ?? 3
            subsamplingConvStride =
                try c.decodeIfPresent(Int.self, forKey: .subsamplingConvStride) ?? 2
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 5000
            scaleInput = try c.decodeIfPresent(Bool.self, forKey: .scaleInput) ?? false
            projectionHiddenSize =
                try c.decodeIfPresent(Int.self, forKey: .projectionHiddenSize) ?? 4096
            projectionBias = try c.decodeIfPresent(Bool.self, forKey: .projectionBias) ?? false
            samplingRate = try c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 16000
            hopLength = try c.decodeIfPresent(Int.self, forKey: .hopLength) ?? 160
            fftLength = try c.decodeIfPresent(Int.self, forKey: .fftLength) ?? 512
            windowLength = try c.decodeIfPresent(Int.self, forKey: .windowLength) ?? 400
            preemphasis = try c.decodeIfPresent(Float.self, forKey: .preemphasis) ?? 0.97
        }

        public var subsamplingLayerCount: Int {
            max(1, Int(log2(Double(max(subsamplingFactor, 1)))))
        }
    }

    public let textConfiguration: TextConfiguration
    public let languageConfiguration: NemotronHConfiguration?
    public let visionConfiguration: VisionConfiguration
    public let soundConfiguration: SoundConfiguration?
    public let modelType: String
    public let downsampleRatio: Float
    public let projectorHiddenSize: Int
    public let vitHiddenSize: Int
    public let visionFinalLayerNorm: Bool
    public let videoPruningRate: Float
    public let videoTemporalPatchSize: Int
    public let imageContextTokenID: Int?
    public let videoContextTokenID: Int?
    public let soundContextTokenID: Int?

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case languageConfiguration = "llm_config"
        case visionConfiguration = "vision_config"
        case soundConfiguration = "sound_config"
        case modelType = "model_type"
        case downsampleRatio = "downsample_ratio"
        case projectorHiddenSize = "projector_hidden_size"
        case vitHiddenSize = "vit_hidden_size"
        case visionFinalLayerNorm = "vision_final_layernorm"
        case videoPruningRate = "video_pruning_rate"
        case videoTemporalPatchSize = "video_temporal_patch_size"
        case imageContextTokenID = "img_context_token_id"
        case videoContextTokenID = "video_context_token_id"
        case soundContextTokenID = "sound_context_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        languageConfiguration =
            (try? c.decode(NemotronHConfiguration.self, forKey: .textConfiguration))
            ?? (try? c.decode(NemotronHConfiguration.self, forKey: .languageConfiguration))
        textConfiguration =
            try languageConfiguration.map { TextConfiguration(hiddenSize: $0.hiddenSize) }
            ?? c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration)
            ?? c.decodeIfPresent(TextConfiguration.self, forKey: .languageConfiguration)
            ?? .init()
        visionConfiguration =
            try c.decodeIfPresent(VisionConfiguration.self, forKey: .visionConfiguration)
            ?? .init()
        soundConfiguration = try c.decodeIfPresent(
            SoundConfiguration.self,
            forKey: .soundConfiguration,
        )
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType)
            ?? "nemotron_h_nano_omni"
        downsampleRatio =
            try c.decodeIfPresent(Float.self, forKey: .downsampleRatio)
            ?? visionConfiguration.downsampleRatio
        projectorHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .projectorHiddenSize)
            ?? textConfiguration.hiddenSize
        vitHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .vitHiddenSize)
            ?? visionConfiguration.hiddenSize
        visionFinalLayerNorm =
            try c.decodeIfPresent(Bool.self, forKey: .visionFinalLayerNorm)
            ?? ((languageConfiguration?.numNextNPredictLayers ?? 0) > 0)
        videoPruningRate = try c.decodeIfPresent(Float.self, forKey: .videoPruningRate) ?? 0
        videoTemporalPatchSize =
            try c.decodeIfPresent(Int.self, forKey: .videoTemporalPatchSize)
            ?? visionConfiguration.videoTemporalPatchSize
        imageContextTokenID = try c.decodeIfPresent(Int.self, forKey: .imageContextTokenID)
        videoContextTokenID = try c.decodeIfPresent(Int.self, forKey: .videoContextTokenID)
        soundContextTokenID = try c.decodeIfPresent(Int.self, forKey: .soundContextTokenID)
    }

    public init(
        textConfiguration: TextConfiguration = .init(),
        languageConfiguration: NemotronHConfiguration? = nil,
        visionConfiguration: VisionConfiguration = .init(),
        soundConfiguration: SoundConfiguration? = nil,
        modelType: String = "nemotron_h_nano_omni",
        downsampleRatio: Float = 0.5,
        projectorHiddenSize: Int? = nil,
        vitHiddenSize: Int? = nil,
        visionFinalLayerNorm: Bool? = nil,
        videoPruningRate: Float = 0,
        videoTemporalPatchSize: Int? = nil,
        imageContextTokenID: Int? = nil,
        videoContextTokenID: Int? = nil,
        soundContextTokenID: Int? = nil,
    ) {
        self.languageConfiguration = languageConfiguration
        self.textConfiguration =
            languageConfiguration.map { TextConfiguration(hiddenSize: $0.hiddenSize) }
            ?? textConfiguration
        self.visionConfiguration = visionConfiguration
        self.soundConfiguration = soundConfiguration
        self.modelType = modelType
        self.downsampleRatio = downsampleRatio
        self.projectorHiddenSize = projectorHiddenSize ?? self.textConfiguration.hiddenSize
        self.vitHiddenSize = vitHiddenSize ?? visionConfiguration.hiddenSize
        self.visionFinalLayerNorm =
            visionFinalLayerNorm
            ?? ((languageConfiguration?.numNextNPredictLayers ?? 0) > 0)
        self.videoPruningRate = videoPruningRate
        self.videoTemporalPatchSize =
            videoTemporalPatchSize ?? visionConfiguration.videoTemporalPatchSize
        self.imageContextTokenID = imageContextTokenID
        self.videoContextTokenID = videoContextTokenID
        self.soundContextTokenID = soundContextTokenID
    }
}

// MARK: - Processor

public struct NemotronOmniProcessorConfiguration: Decodable, Sendable {
    public let processorClass: String
    public let imageToken: String
    public let imageStartToken: String
    public let imageEndToken: String
    public let videoToken: String
    public let audioToken: String
    public let audioStartToken: String
    public let audioEndToken: String
    public let imageTokenID: Int?
    public let videoTokenID: Int?
    public let audioTokenID: Int?
    public let audioStartTokenID: Int?
    public let audioEndTokenID: Int?
    public let normMean: [CGFloat]
    public let normStd: [CGFloat]
    public let patchSize: Int
    public let downsampleRatio: Float
    public let minNumPatches: Int
    public let maxNumPatches: Int
    public let maxModelLength: Int
    public let videoTargetNumPatches: Int
    public let videoMaintainAspectRatio: Bool
    public let videoTemporalPatchSize: Int
    public let videoFPS: Double
    public let maxVideoFrames: Int
    public let audioSamplingRate: Int
    public let audioSubsamplingFactor: Int
    public let audioHopLength: Int
    public let audioSubsamplingConvKernelSize: Int
    public let audioSubsamplingConvStride: Int
    public let soundConfiguration: NemotronOmniConfiguration.SoundConfiguration

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageToken = "image_token"
        case imageStartToken = "image_start_token"
        case imageEndToken = "image_end_token"
        case videoToken = "video_token"
        case audioToken = "audio_token"
        case audioStartToken = "audio_start_token"
        case audioEndToken = "audio_end_token"
        case imageTokenID = "image_token_id"
        case imageContextTokenID = "img_context_token_id"
        case imageTokenIndex = "image_token_index"
        case videoTokenID = "video_token_id"
        case videoContextTokenID = "video_context_token_id"
        case audioTokenID = "audio_token_id"
        case audioStartTokenID = "audio_start_token_id"
        case audioEndTokenID = "audio_end_token_id"
        case soundContextTokenID = "sound_context_token_id"
        case imageProcessor = "image_processor"
        case preprocessorNormMean = "norm_mean"
        case preprocessorNormStd = "norm_std"
        case patchSize = "patch_size"
        case downsampleRatio = "downsample_ratio"
        case minNumPatches = "min_num_patches"
        case maxNumPatches = "max_num_patches"
        case maxModelLength = "max_model_len"
        case videoTargetNumPatches = "video_target_num_patches"
        case videoMaintainAspectRatio = "video_maintain_aspect_ratio"
        case videoTemporalPatchSize = "video_temporal_patch_dim"
        case videoFPS = "video_fps"
        case fps
        case maxVideoFrames = "max_num_frames"
        case maxFrames = "max_frames"
        case numFrames = "num_frames"
        case audioSamplingRate = "audio_sampling_rate"
        case audioSubsamplingFactor = "audio_subsampling_factor"
        case audioHopLength = "audio_hop_length"
        case audioSubsamplingConvKernelSize = "audio_subsampling_conv_kernel_size"
        case audioSubsamplingConvStride = "audio_subsampling_conv_stride"
        case soundConfiguration = "sound_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let imageProcessor =
            try? c.nestedContainer(keyedBy: CodingKeys.self, forKey: .imageProcessor)
        processorClass =
            try c.decodeIfPresent(String.self, forKey: .processorClass)
            ?? "NemotronHNanoOmniProcessor"
        imageToken = try c.decodeIfPresent(String.self, forKey: .imageToken) ?? "<image>"
        imageStartToken =
            try c.decodeIfPresent(String.self, forKey: .imageStartToken) ?? "<img>"
        imageEndToken =
            try c.decodeIfPresent(String.self, forKey: .imageEndToken) ?? "</img>"
        videoToken = try c.decodeIfPresent(String.self, forKey: .videoToken) ?? "<video>"
        audioToken = try c.decodeIfPresent(String.self, forKey: .audioToken) ?? "<so_embedding>"
        audioStartToken =
            try c.decodeIfPresent(String.self, forKey: .audioStartToken) ?? "<so_start>"
        audioEndToken =
            try c.decodeIfPresent(String.self, forKey: .audioEndToken) ?? "<so_end>"
        imageTokenID =
            try c.decodeIfPresent(Int.self, forKey: .imageTokenID)
            ?? c.decodeIfPresent(Int.self, forKey: .imageContextTokenID)
            ?? c.decodeIfPresent(Int.self, forKey: .imageTokenIndex)
        videoTokenID =
            try c.decodeIfPresent(Int.self, forKey: .videoTokenID)
            ?? c.decodeIfPresent(Int.self, forKey: .videoContextTokenID)
        audioTokenID =
            try c.decodeIfPresent(Int.self, forKey: .audioTokenID)
            ?? c.decodeIfPresent(Int.self, forKey: .soundContextTokenID)
        audioStartTokenID = try c.decodeIfPresent(Int.self, forKey: .audioStartTokenID)
        audioEndTokenID = try c.decodeIfPresent(Int.self, forKey: .audioEndTokenID)

        let baseSoundConfiguration =
            try c.decodeIfPresent(
                NemotronOmniConfiguration.SoundConfiguration.self,
                forKey: .soundConfiguration,
            )
            ?? NemotronOmniConfiguration.SoundConfiguration()

        normMean =
            try imageProcessor?.decodeIfPresent([CGFloat].self, forKey: .preprocessorNormMean)
            ?? c.decodeIfPresent([CGFloat].self, forKey: .preprocessorNormMean)
            ?? [0.5, 0.5, 0.5]
        normStd =
            try imageProcessor?.decodeIfPresent([CGFloat].self, forKey: .preprocessorNormStd)
            ?? c.decodeIfPresent([CGFloat].self, forKey: .preprocessorNormStd)
            ?? [0.5, 0.5, 0.5]
        patchSize =
            try imageProcessor?.decodeIfPresent(Int.self, forKey: .patchSize)
            ?? c.decodeIfPresent(Int.self, forKey: .patchSize)
            ?? 16
        downsampleRatio =
            try imageProcessor?.decodeIfPresent(Float.self, forKey: .downsampleRatio)
            ?? c.decodeIfPresent(Float.self, forKey: .downsampleRatio)
            ?? 0.5
        minNumPatches =
            try imageProcessor?.decodeIfPresent(Int.self, forKey: .minNumPatches)
            ?? c.decodeIfPresent(Int.self, forKey: .minNumPatches)
            ?? 1024
        maxNumPatches =
            try imageProcessor?.decodeIfPresent(Int.self, forKey: .maxNumPatches)
            ?? c.decodeIfPresent(Int.self, forKey: .maxNumPatches)
            ?? 13312
        maxModelLength =
            try imageProcessor?.decodeIfPresent(Int.self, forKey: .maxModelLength)
            ?? c.decodeIfPresent(Int.self, forKey: .maxModelLength)
            ?? 16384
        videoTargetNumPatches =
            try imageProcessor?.decodeIfPresent(Int.self, forKey: .videoTargetNumPatches)
            ?? c.decodeIfPresent(Int.self, forKey: .videoTargetNumPatches)
            ?? 1024
        videoMaintainAspectRatio =
            try imageProcessor?.decodeIfPresent(Bool.self, forKey: .videoMaintainAspectRatio)
            ?? c.decodeIfPresent(Bool.self, forKey: .videoMaintainAspectRatio)
            ?? true
        videoTemporalPatchSize =
            try c.decodeIfPresent(Int.self, forKey: .videoTemporalPatchSize) ?? 2
        videoFPS =
            try c.decodeIfPresent(Double.self, forKey: .videoFPS)
            ?? c.decodeIfPresent(Double.self, forKey: .fps)
            ?? 2
        maxVideoFrames =
            try c.decodeIfPresent(Int.self, forKey: .maxVideoFrames)
            ?? c.decodeIfPresent(Int.self, forKey: .maxFrames)
            ?? c.decodeIfPresent(Int.self, forKey: .numFrames)
            ?? Int.max

        audioSamplingRate =
            try c.decodeIfPresent(Int.self, forKey: .audioSamplingRate)
            ?? baseSoundConfiguration.samplingRate
        audioSubsamplingFactor =
            try c.decodeIfPresent(Int.self, forKey: .audioSubsamplingFactor)
            ?? baseSoundConfiguration.subsamplingFactor
        audioHopLength =
            try c.decodeIfPresent(Int.self, forKey: .audioHopLength)
            ?? baseSoundConfiguration.hopLength
        audioSubsamplingConvKernelSize =
            try c.decodeIfPresent(Int.self, forKey: .audioSubsamplingConvKernelSize)
            ?? baseSoundConfiguration.subsamplingConvKernelSize
        audioSubsamplingConvStride =
            try c.decodeIfPresent(Int.self, forKey: .audioSubsamplingConvStride)
            ?? baseSoundConfiguration.subsamplingConvStride

        soundConfiguration = NemotronOmniConfiguration.SoundConfiguration(
            modelType: baseSoundConfiguration.modelType,
            hiddenSize: baseSoundConfiguration.hiddenSize,
            numAttentionHeads: baseSoundConfiguration.numAttentionHeads,
            numHiddenLayers: baseSoundConfiguration.numHiddenLayers,
            intermediateSize: baseSoundConfiguration.intermediateSize,
            hiddenActivation: baseSoundConfiguration.hiddenActivation,
            attentionBias: baseSoundConfiguration.attentionBias,
            convolutionBias: baseSoundConfiguration.convolutionBias,
            convKernelSize: baseSoundConfiguration.convKernelSize,
            subsamplingFactor: audioSubsamplingFactor,
            subsamplingConvChannels: baseSoundConfiguration.subsamplingConvChannels,
            numMelBins: baseSoundConfiguration.numMelBins,
            featureSize: baseSoundConfiguration.featureSize,
            subsamplingConvKernelSize: audioSubsamplingConvKernelSize,
            subsamplingConvStride: audioSubsamplingConvStride,
            maxPositionEmbeddings: baseSoundConfiguration.maxPositionEmbeddings,
            scaleInput: baseSoundConfiguration.scaleInput,
            projectionHiddenSize: baseSoundConfiguration.projectionHiddenSize,
            projectionBias: baseSoundConfiguration.projectionBias,
            samplingRate: audioSamplingRate,
            hopLength: audioHopLength,
            fftLength: baseSoundConfiguration.fftLength,
            windowLength: baseSoundConfiguration.windowLength,
            preemphasis: baseSoundConfiguration.preemphasis,
        )
    }
}

public struct NemotronOmniProcessor: UserInputProcessor {
    private let config: NemotronOmniProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageTokenID: Int?
    private let videoTokenID: Int?
    private let imageStartTokenID: Int?
    private let imageEndTokenID: Int?
    private let audioTokenID: Int?
    private let audioStartTokenID: Int?
    private let audioEndTokenID: Int?
    private let soundFeatureExtractor: NemotronOmniAudio.SoundFeatureExtractor

    public init(_ config: NemotronOmniProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        imageTokenID = tokenizer.convertTokenToId(config.imageToken) ?? config.imageTokenID
        videoTokenID = tokenizer.convertTokenToId(config.videoToken) ?? config.videoTokenID
        imageStartTokenID =
            tokenizer.convertTokenToId(config.imageStartToken)
            ?? tokenizer.encode(text: config.imageStartToken, addSpecialTokens: false).first
        imageEndTokenID =
            tokenizer.convertTokenToId(config.imageEndToken)
            ?? tokenizer.encode(text: config.imageEndToken, addSpecialTokens: false).first
        audioTokenID = tokenizer.convertTokenToId(config.audioToken) ?? config.audioTokenID
        audioStartTokenID =
            tokenizer.convertTokenToId(config.audioStartToken) ?? config.audioStartTokenID
        audioEndTokenID =
            tokenizer.convertTokenToId(config.audioEndToken) ?? config.audioEndTokenID
        soundFeatureExtractor = NemotronOmniAudio.SoundFeatureExtractor(config.soundConfiguration)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = NemotronOmniMessageGenerator().generate(from: input)
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext,
        )

        let images = input.images
        let videos = input.videos
        var processedImage: LMInput.ProcessedImage?
        var imageTokenCounts: [Int] = []
        if !images.isEmpty {
            let processed = try images.map {
                try preprocess(image: $0.asCIImage(), processing: input.processing, video: false)
            }
            imageTokenCounts = processed.map(\.tokenCount)
            processedImage = LMInput.ProcessedImage(
                pixels: Self.stackAndPad(processed.map(\.pixels)),
                frames: processed.map(\.frame),
            )
            promptTokens = try expandVisionTokens(
                in: promptTokens,
                markerTokenID: requiredImageTokenID(),
                tokenCounts: imageTokenCounts,
            )
        }

        var processedVideo: LMInput.ProcessedVideo?
        var videoTokenCounts: [Int] = []
        if !videos.isEmpty {
            var pixels: [MLXArray] = []
            var frames: [THW] = []
            var counts: [Int] = []
            for video in videos {
                let processed = try await preprocess(video: video, processing: input.processing)
                pixels.append(contentsOf: processed.pixels)
                frames.append(contentsOf: processed.frames)
                counts.append(processed.tokenCount)
            }
            videoTokenCounts = counts
            processedVideo = LMInput.ProcessedVideo(
                pixels: Self.stackAndPad(pixels),
                frames: frames,
            )
            promptTokens = try expandVisionTokens(
                in: promptTokens,
                markerTokenID: requiredVideoTokenID(),
                tokenCounts: videoTokenCounts,
            )
        }

        var processedAudio: LMInput.ProcessedAudio?
        if !input.audios.isEmpty {
            let audioData = try await preprocess(audios: input.audios, processing: input.processing)
            processedAudio = audioData.audio
            promptTokens = try expandAudioTokens(
                in: promptTokens,
                tokenCounts: audioData.tokenCounts,
            )
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: promptArray, mask: ones(like: promptArray).asType(.int8)),
            image: processedImage,
            video: processedVideo,
            audio: processedAudio,
        )
    }

    private func requiredImageTokenID() throws -> Int {
        guard let imageTokenID else {
            throw VLMError.processing("Nemotron Omni image token ID is unavailable.")
        }
        return imageTokenID
    }

    private func requiredVideoTokenID() throws -> Int {
        guard let videoTokenID else {
            throw VLMError.processing("Nemotron Omni video token ID is unavailable.")
        }
        return videoTokenID
    }

    private func requiredImageBoundaryTokenIDs() throws -> (start: Int, end: Int) {
        guard let imageStartTokenID else {
            throw VLMError.processing("Nemotron Omni image start token ID is unavailable.")
        }
        guard let imageEndTokenID else {
            throw VLMError.processing("Nemotron Omni image end token ID is unavailable.")
        }
        return (imageStartTokenID, imageEndTokenID)
    }

    private func expandVisionTokens(
        in tokens: [Int],
        markerTokenID: Int,
        tokenCounts: [Int],
    )
        throws -> [Int]
    {
        guard !tokenCounts.isEmpty else {
            return tokens
        }
        let imageTokenID = try requiredImageTokenID()
        let boundary = try requiredImageBoundaryTokenIDs()
        var mediaIndex = 0
        var expanded: [Int] = []
        expanded.reserveCapacity(tokens.count + tokenCounts.reduce(0, +) + 2 * tokenCounts.count)

        for token in tokens {
            if token == markerTokenID {
                guard mediaIndex < tokenCounts.count else {
                    throw VLMError.processing(
                        "Nemotron Omni media prompt tokens do not match inputs.")
                }
                expanded.append(boundary.start)
                expanded.append(
                    contentsOf: Array(repeating: imageTokenID, count: tokenCounts[mediaIndex]))
                expanded.append(boundary.end)
                mediaIndex += 1
            } else {
                expanded.append(token)
            }
        }

        guard mediaIndex == tokenCounts.count else {
            throw VLMError.processing("Nemotron Omni media inputs require matching prompt tokens.")
        }
        return expanded
    }

    private struct ProcessedVision {
        let pixels: MLXArray
        let frame: THW
        let tokenCount: Int
    }

    private func preprocess(image: CIImage, processing: UserInput.Processing?, video: Bool)
        throws -> ProcessedVision
    {
        let source = MediaProcessing.apply(image, processing: processing).toSRGB()
        let targetPatches =
            video
            ? targetPatchesForVideo(size: source.extent.size)
            : targetPatchesForImage(size: source.extent.size)
        let targetWidth = targetPatches.width * config.patchSize
        let targetHeight = targetPatches.height * config.patchSize
        var processed = MediaProcessing.resampleBicubic(
            source,
            to: CGSize(width: targetWidth, height: targetHeight),
        )
        processed = MediaProcessing.normalize(
            processed,
            mean: normMeanTuple,
            std: normStdTuple,
        )
        let pixels = MediaProcessing.asMLXArray(processed)
        let downsample = max(1, Int(round(1 / Double(max(config.downsampleRatio, 0.0001)))))
        let tokenCount = max(
            1, (targetPatches.width * targetPatches.height) / (downsample * downsample))
        return .init(
            pixels: pixels,
            frame: THW(1, targetHeight, targetWidth),
            tokenCount: tokenCount,
        )
    }

    private func preprocess(video: UserInput.Video, processing: UserInput.Processing?) async throws
        -> (pixels: [MLXArray], frames: [THW], tokenCount: Int)
    {
        let maxFrames = max(config.maxVideoFrames, 1)
        let sequence = try await MediaProcessing.asProcessedSequence(
            video,
            targetFPS: { _ in config.videoFPS > 0 ? config.videoFPS : 2 },
            maxFrames: maxFrames,
        ) { frame in
            let source = try MediaProcessing.apply(
                frame.image.asCIImage(), processing: processing
            ).toSRGB()
            let targetPatches = targetPatchesForVideo(size: source.extent.size)
            let targetWidth = targetPatches.width * config.patchSize
            let targetHeight = targetPatches.height * config.patchSize
            var processed = MediaProcessing.resampleBicubic(
                source,
                to: CGSize(width: targetWidth, height: targetHeight),
            )
            processed = MediaProcessing.normalize(
                processed,
                mean: normMeanTuple,
                std: normStdTuple,
            )
            return UserInput.VideoFrame(image: .ciImage(processed), timeStamp: frame.timeStamp)
        }
        let frameArrays = Array(sequence.frames.prefix(maxFrames))
        guard !frameArrays.isEmpty else {
            throw VLMError.processing("Nemotron Omni video preprocessing produced no frames.")
        }

        var processed: [ProcessedVision] = []
        processed.reserveCapacity(frameArrays.count)
        let downsample = max(1, Int(round(1 / Double(max(config.downsampleRatio, 0.0001)))))
        for frame in frameArrays {
            let patchesWide = max(frame.dim(-1) / config.patchSize, 1)
            let patchesHigh = max(frame.dim(-2) / config.patchSize, 1)
            processed.append(
                .init(
                    pixels: frame,
                    frame: THW(1, frame.dim(-2), frame.dim(-1)),
                    tokenCount: max(1, (patchesWide * patchesHigh) / (downsample * downsample)),
                ))
        }
        let temporal = max(config.videoTemporalPatchSize, 1)
        let tokensPerTubelet = processed.first?.tokenCount ?? 1
        let groups = (processed.count + temporal - 1) / temporal
        return (
            pixels: processed.map(\.pixels),
            frames: processed.map(\.frame),
            tokenCount: tokensPerTubelet * groups,
        )
    }

    private var normMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (
            config.normMean.indices.contains(0) ? config.normMean[0] : 0.5,
            config.normMean.indices.contains(1) ? config.normMean[1] : 0.5,
            config.normMean.indices.contains(2) ? config.normMean[2] : 0.5,
        )
    }

    private var normStdTuple: (CGFloat, CGFloat, CGFloat) {
        (
            config.normStd.indices.contains(0) ? config.normStd[0] : 0.5,
            config.normStd.indices.contains(1) ? config.normStd[1] : 0.5,
            config.normStd.indices.contains(2) ? config.normStd[2] : 0.5,
        )
    }

    private func targetPatchesForImage(size: CGSize) -> (width: Int, height: Int) {
        let numTokensAvailable = max(config.maxModelLength - 4, 1)
        let downsample = max(1, Int(round(1 / Double(max(config.downsampleRatio, 0.0001)))))
        var budget = numTokensAvailable * downsample * downsample
        budget = max(budget, config.minNumPatches)
        budget = max(min(budget, config.maxNumPatches), config.minNumPatches)
        return targetPatches(size: size, tokensAvailable: budget)
    }

    private func targetPatchesForVideo(size: CGSize) -> (width: Int, height: Int) {
        let target = max(config.videoTargetNumPatches, 1)
        let divisor = max(1, Int(round(1 / Double(max(config.downsampleRatio, 0.0001)))))
        guard config.videoMaintainAspectRatio else {
            let side = max(divisor, Int(sqrt(Double(target))) / divisor * divisor)
            return (side, side)
        }

        let aspect = Double(size.width / max(size.height, 1))
        var height = max(Int(round(sqrt(Double(target) / max(aspect, 0.0001)))), 1)
        var width = max(Int(round(sqrt(Double(target) * max(aspect, 0.0001)))), 1)
        if divisor > 1 {
            let heightRemainder = height % divisor
            let widthRemainder = width % divisor
            let heightUp = height + (heightRemainder == 0 ? 0 : divisor - heightRemainder)
            let widthUp = width + (widthRemainder == 0 ? 0 : divisor - widthRemainder)
            if heightUp * widthUp <= target {
                height = heightUp
                width = widthUp
            } else {
                height = max(divisor, height - heightRemainder)
                width = max(divisor, width - widthRemainder)
            }
        }
        return (width, height)
    }

    private func targetPatches(size: CGSize, tokensAvailable: Int) -> (width: Int, height: Int) {
        let originalWidth = max(Int(size.width.rounded()), 1)
        let originalHeight = max(Int(size.height.rounded()), 1)
        var height = Int((Double(originalHeight) / Double(config.patchSize) + 0.5).rounded())
        var width = Int((Double(originalWidth) / Double(config.patchSize) + 0.5).rounded())
        height = max(height, 1)
        width = max(width, 1)
        let patches = max(height * width, 1)
        let factor = min(sqrt(Double(tokensAvailable) / Double(patches)), 1)
        height = max(Int(floor(factor * Double(height))), 1)
        width = max(Int(floor(factor * Double(width))), 1)

        if tokensAvailable > config.minNumPatches, height * width < config.minNumPatches {
            let up = sqrt(Double(config.minNumPatches) / Double(max(height * width, 1)))
            height = max(Int(ceil(up * Double(height))), 1)
            width = max(Int(ceil(up * Double(width))), 1)
        }

        let divisor = max(1, Int(round(1 / Double(max(config.downsampleRatio, 0.0001)))))
        let heightRemainder = height % divisor
        if heightRemainder != 0 {
            let increment = divisor - heightRemainder
            if (height + increment) * width <= tokensAvailable {
                height += increment
            } else {
                height = max(divisor, height - heightRemainder)
            }
        }
        let widthRemainder = width % divisor
        if widthRemainder != 0 {
            let increment = divisor - widthRemainder
            if height * (width + increment) <= tokensAvailable {
                width += increment
            } else {
                width = max(divisor, width - widthRemainder)
            }
        }
        return (width, height)
    }

    private static func stackAndPad(_ values: [MLXArray]) -> MLXArray {
        guard let maxHeight = values.map({ $0.dim(-2) }).max(),
            let maxWidth = values.map({ $0.dim(-1) }).max()
        else {
            return MLXArray.zeros([0], dtype: .float32)
        }
        let paddedValues = values.map { value in
            padded(
                value,
                widths: [
                    0,
                    0,
                    .init((0, maxHeight - value.dim(-2))),
                    .init((0, maxWidth - value.dim(-1))),
                ],
            )
        }
        return concatenated(paddedValues, axis: 0)
    }

    public func preprocess(audios: [UserInput.Audio], processing: UserInput.Processing) async throws
        -> (audio: LMInput.ProcessedAudio, tokenCounts: [Int])
    {
        var audioProcessing = processing.audio
        audioProcessing.sampleRate = Double(config.audioSamplingRate)
        audioProcessing.channels = 1

        var samples: [MLXArray] = []
        samples.reserveCapacity(audios.count)
        for audio in audios {
            try await samples.append(audio.asMLXArray(processing: audioProcessing))
        }

        let monoSamples = samples.map { Gemma4AudioFeatureExtractor.monoFloatSamples($0) }
        let tokenCounts = monoSamples.map { sample in
            max(1, Self.estimatedAudioTokenCount(sampleCount: sample.dim(0), config: config))
        }
        let extracted = try soundFeatureExtractor(monoSamples)
        let metadata = Self.packMetadata(
            tokenCounts: tokenCounts,
            fullLengths: extracted.fullLengths,
            attentionMask: extracted.attentionMask,
        )

        return (
            audio: LMInput.ProcessedAudio(features: extracted.inputFeatures, mask: metadata),
            tokenCounts: tokenCounts,
        )
    }

    private func expandAudioTokens(in tokens: [Int], tokenCounts: [Int]) throws -> [Int] {
        guard let audioTokenID else {
            throw VLMError.processing("Nemotron Omni audio token ID is unavailable.")
        }
        guard let audioStartTokenID else {
            throw VLMError.processing("Nemotron Omni audio start token ID is unavailable.")
        }
        guard let audioEndTokenID else {
            throw VLMError.processing("Nemotron Omni audio end token ID is unavailable.")
        }

        var audioIndex = 0
        var expandedTokens: [Int] = []
        expandedTokens.reserveCapacity(
            tokens.count + tokenCounts.reduce(0, +) + 2 * tokenCounts.count)

        for token in tokens {
            if token == audioTokenID {
                guard audioIndex < tokenCounts.count else {
                    throw VLMError.processing("Audio prompt tokens do not match audio inputs.")
                }
                expandedTokens.append(audioStartTokenID)
                expandedTokens.append(
                    contentsOf: Array(repeating: audioTokenID, count: tokenCounts[audioIndex]),
                )
                expandedTokens.append(audioEndTokenID)
                audioIndex += 1
            } else {
                expandedTokens.append(token)
            }
        }

        guard audioIndex == tokenCounts.count else {
            throw VLMError.processing("Audio prompt tokens do not match audio inputs.")
        }
        return expandedTokens
    }

    static func estimatedAudioTokenCount(
        sampleCount: Int,
        config: NemotronOmniProcessorConfiguration,
    )
        -> Int
    {
        var length = 1 + sampleCount / max(config.audioHopLength, 1)
        let kernelSize = config.audioSubsamplingConvKernelSize
        let stride = max(config.audioSubsamplingConvStride, 1)
        let layerCount = max(0, Int(log2(Double(max(config.audioSubsamplingFactor, 1)))))
        let addPadding = ((kernelSize - 1) / 2) * 2 - kernelSize
        for _ in 0 ..< layerCount {
            length = (length + addPadding) / stride + 1
        }
        return max(length, 1)
    }

    private static func packMetadata(
        tokenCounts: [Int],
        fullLengths: MLXArray,
        attentionMask: MLXArray,
    )
        -> MLXArray
    {
        let batchSize = tokenCounts.count
        let featureLength = attentionMask.dim(1)
        let fullLengthValues = fullLengths.asArray(Int32.self)
        let attentionValues = attentionMask.asArray(Int32.self)
        var values: [Int32] = []
        values.reserveCapacity(batchSize * (featureLength + 2))

        for batchIndex in 0 ..< batchSize {
            values.append(Int32(tokenCounts[batchIndex]))
            values.append(fullLengthValues[batchIndex])
            let rowStart = batchIndex * featureLength
            values.append(contentsOf: attentionValues[rowStart ..< rowStart + featureLength])
        }

        return MLXArray(values, [batchSize, featureLength + 2])
    }
}

public struct NemotronOmniMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> Message {
        var dictionary: Message =
            if message.role == .system {
                [
                    "role": message.role.rawValue,
                    "content": message.content,
                ]
            } else {
                [
                    "role": message.role.rawValue,
                    "content": message.images.map { _ in ["type": "image"] }
                        + message.videos.map { _ in ["type": "video"] }
                        + message.audios.map { _ in ["type": "audio"] }
                        + [["type": "text", "text": message.content]],
                ]
            }
        addToolMetadata(to: &dictionary, for: message)
        return dictionary
    }
}

// MARK: - Vision

private enum NemotronOmniVision {
    struct Output {
        let summary: MLXArray
        let features: MLXArray
    }

    final class SquaredReLU: Module, UnaryLayer {
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            nemotronOmniReluSquared(x)
        }
    }

    final class InputConditioner: Module, UnaryLayer {
        @ModuleInfo(key: "norm_mean") var normMean: MLXArray
        @ModuleInfo(key: "norm_std") var normStd: MLXArray

        override init() {
            _normMean.wrappedValue = MLXArray.zeros([3, 1, 1])
            _normStd.wrappedValue = MLXArray.ones([3, 1, 1])
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            (x - normMean.asType(x.dtype)) / normStd.asType(x.dtype)
        }
    }

    final class ClsToken: Module, UnaryLayer {
        let numTokens: Int
        let numRegisters: Int

        @ModuleInfo(key: "token") var token: MLXArray

        init(embedDimensions: Int, numTokens: Int = 1, numRegisters: Int = 0) {
            self.numTokens = numTokens
            self.numRegisters = numRegisters
            _token.wrappedValue = MLXArray.zeros([numTokens + numRegisters, embedDimensions])
            super.init()
        }

        var numPatches: Int {
            numTokens + numRegisters
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let tokens = broadcast(
                token.asType(x.dtype).expandedDimensions(axis: 0),
                to: [x.dim(0), token.dim(0), token.dim(1)],
            )
            return concatenated([tokens, x], axis: 1)
        }
    }

    final class ViTPatchGenerator: Module {
        let patchSize: Int
        let embedDimensions: Int
        let numRows: Int
        let numColumns: Int
        let inputDimensions: (Int, Int)
        let cpeMode: Bool

        @ModuleInfo(key: "cls_token") var clsToken: ClsToken
        @ModuleInfo var embedder: Linear
        @ModuleInfo(key: "video_embedder") var videoEmbedder: Linear
        @ModuleInfo(key: "pos_embed") var posEmbed: MLXArray

        init(_ config: NemotronOmniConfiguration.VisionConfiguration) {
            patchSize = config.patchSize
            embedDimensions = config.hiddenSize
            numRows = max(config.maxResolution / config.patchSize, 1)
            numColumns = max(config.maxResolution / config.patchSize, 1)
            inputDimensions = (
                max(config.imageSize / config.patchSize, 1),
                max(config.imageSize / config.patchSize, 1),
            )
            cpeMode = (numRows, numColumns) != inputDimensions
            _clsToken.wrappedValue = ClsToken(
                embedDimensions: config.hiddenSize, numTokens: config.numClassTokens,
                numRegisters: config.numRegisters)
            _embedder.wrappedValue = Linear(
                3 * config.patchSize * config.patchSize,
                config.hiddenSize,
                bias: false,
            )
            _videoEmbedder.wrappedValue = Linear(
                max(config.videoTemporalPatchSize, 1) * 3 * config.patchSize * config.patchSize,
                config.hiddenSize,
                bias: false,
            )
            _posEmbed.wrappedValue = MLXArray.zeros([1, numRows * numColumns, config.hiddenSize])
            super.init()
        }

        var numClassTokens: Int {
            clsToken.numTokens
        }

        var numRegisters: Int {
            clsToken.numRegisters
        }

        var numSkip: Int {
            numClassTokens + numRegisters
        }

        func callAsFunction(_ x: MLXArray, useVideoEmbedder: Bool = false) -> MLXArray {
            var patches = imageToPatches(x)
            patches = useVideoEmbedder ? videoEmbedder(patches) : embedder(patches)
            let inputRows = max(x.dim(-2) / patchSize, 1)
            let inputColumns = max(x.dim(-1) / patchSize, 1)
            patches =
                patches
                + positionalEmbeddings(
                    batchSize: x.dim(0),
                    inputRows: inputRows,
                    inputColumns: inputColumns,
                    dtype: patches.dtype,
                )
            return clsToken(patches)
        }

        private func imageToPatches(_ x: MLXArray) -> MLXArray {
            let batch = x.dim(0)
            let channels = x.dim(1)
            let patchRows = max(x.dim(2) / patchSize, 1)
            let patchColumns = max(x.dim(3) / patchSize, 1)
            return
                x
                .reshaped(batch, channels, patchRows, patchSize, patchColumns, patchSize)
                .transposed(0, 2, 4, 1, 3, 5)
                .reshaped(batch, patchRows * patchColumns, channels * patchSize * patchSize)
        }

        private func positionalEmbeddings(
            batchSize: Int,
            inputRows: Int,
            inputColumns: Int,
            dtype: DType,
        )
            -> MLXArray
        {
            let pos: MLXArray
            if numRows == inputRows, numColumns == inputColumns {
                pos = posEmbed
            } else {
                var grid =
                    posEmbed
                    .reshaped(1, numRows, numColumns, embedDimensions)
                    .transposed(0, 3, 1, 2)
                if cpeMode {
                    let maxDimension = max(inputRows, inputColumns)
                    grid = bicubicInterpolate(
                        grid,
                        size: (maxDimension, maxDimension),
                        alignCorners: false,
                        antialias: false,
                    )
                    grid = grid[0..., 0..., ..<inputRows, ..<inputColumns]
                }
                if grid.dim(2) != inputRows || grid.dim(3) != inputColumns {
                    grid = bicubicInterpolate(
                        grid,
                        size: (inputRows, inputColumns),
                        alignCorners: false,
                        antialias: false,
                    )
                }
                pos =
                    grid
                    .transposed(0, 2, 3, 1)
                    .reshaped(1, inputRows * inputColumns, embedDimensions)
            }
            return broadcast(pos.asType(dtype), to: [batchSize, pos.dim(1), pos.dim(2)])
        }
    }

    final class Attention: Module {
        let heads: Int
        let headDimensions: Int
        let scale: Float

        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear

        init(dimensions: Int, heads: Int) {
            self.heads = heads
            headDimensions = dimensions / heads
            scale = pow(Float(headDimensions), -0.5)
            _qkv.wrappedValue = Linear(dimensions, dimensions * 3, bias: true)
            _proj.wrappedValue = Linear(dimensions, dimensions, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let batch = x.dim(0)
            let length = x.dim(1)
            let dimensions = x.dim(2)
            let states = qkv(x)
                .reshaped(batch, length, 3, heads, headDimensions)
                .transposed(2, 0, 3, 1, 4)
            let output = MLXFast.scaledDotProductAttention(
                queries: states[0],
                keys: states[1],
                values: states[2],
                scale: scale,
                mask: nil,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, dimensions)
            return proj(output)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _fc1.wrappedValue = Linear(dimensions, hiddenDimensions, bias: true)
            _fc2.wrappedValue = Linear(hiddenDimensions, dimensions, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(gelu(fc1(x)))
        }
    }

    final class Block: Module {
        @ModuleInfo var norm1: LayerNorm
        @ModuleInfo var attn: Attention
        @ModuleInfo var norm2: LayerNorm
        @ModuleInfo var mlp: MLP

        init(_ config: NemotronOmniConfiguration.VisionConfiguration) {
            _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _attn.wrappedValue = Attention(
                dimensions: config.hiddenSize,
                heads: config.numAttentionHeads,
            )
            _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _mlp.wrappedValue = MLP(
                dimensions: config.hiddenSize,
                hiddenDimensions: config.intermediateSize,
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let h = x + attn(norm1(x))
            return h + mlp(norm2(h))
        }
    }

    final class RadioBackbone: Module {
        @ModuleInfo(key: "patch_generator") var patchGenerator: ViTPatchGenerator
        @ModuleInfo var blocks: [Block]

        init(_ config: NemotronOmniConfiguration.VisionConfiguration) {
            _patchGenerator.wrappedValue = ViTPatchGenerator(config)
            _blocks.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in Block(config) }
            super.init()
        }

        func forwardFeatures(_ x: MLXArray, useVideoEmbedder: Bool = false) -> MLXArray {
            var hidden = patchGenerator(x, useVideoEmbedder: useVideoEmbedder)
            for block in blocks {
                hidden = block(hidden)
            }
            return hidden
        }
    }

    final class RadioModel: Module {
        @ModuleInfo(key: "input_conditioner") var inputConditioner: InputConditioner
        @ModuleInfo var model: RadioBackbone

        init(_ config: NemotronOmniConfiguration.VisionConfiguration) {
            _inputConditioner.wrappedValue = InputConditioner()
            _model.wrappedValue = RadioBackbone(config)
            super.init()
        }

        func callAsFunction(_ x: MLXArray, useVideoEmbedder: Bool = false) -> Output {
            let hidden = model.forwardFeatures(x, useVideoEmbedder: useVideoEmbedder)
            let generator = model.patchGenerator
            let summary = hidden[0..., ..<generator.numClassTokens, 0...]
                .reshaped(hidden.dim(0), -1)
            let features = hidden[0..., generator.numSkip..., 0...]
            return Output(summary: summary, features: features)
        }
    }

    final class VisionModel: Module {
        @ModuleInfo(key: "radio_model") var radioModel: RadioModel

        init(_ config: NemotronOmniConfiguration.VisionConfiguration) {
            _radioModel.wrappedValue = RadioModel(config)
            super.init()
        }

        func callAsFunction(_ pixelValues: MLXArray, useVideoEmbedder: Bool = false) -> Output {
            radioModel(pixelValues, useVideoEmbedder: useVideoEmbedder)
        }
    }

    final class VisionProjection: Module, UnaryLayer {
        @ModuleInfo(key: "layers") var layers: [Module]

        init(_ config: NemotronOmniConfiguration) {
            let downsampleFactor = config.visionConfiguration.downsampleFactor
            let inputFeatures = config.vitHiddenSize * downsampleFactor * downsampleFactor
            _layers.wrappedValue = [
                RMSNorm(dimensions: inputFeatures, eps: 1e-5),
                Linear(inputFeatures, config.projectorHiddenSize, bias: false),
                SquaredReLU(),
                Linear(
                    config.projectorHiddenSize, config.textConfiguration.hiddenSize, bias: false),
            ]
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var hidden = x
            for layer in layers {
                hidden = (layer as! any UnaryLayer)(hidden)
            }
            return hidden
        }
    }
}

// MARK: - Model

public enum NemotronOmniError: LocalizedError {
    case languageConfigurationMissing
    case soundConfigurationMissing
    case soundContextTokenMissing
    case imageContextTokenMissing
    case audioMetadataMissing
    case soundFeatureCountMismatch(expected: Int, actual: Int)
    case visionFeatureCountMismatch(expected: Int, actual: Int)
    case videoPruningUnsupported

    public var errorDescription: String? {
        switch self {
        case .languageConfigurationMissing:
            "Nemotron Omni requires a Nemotron-H text_config or llm_config."
        case .soundConfigurationMissing:
            "Nemotron Omni audio inputs require sound_config."
        case .soundContextTokenMissing:
            "Nemotron Omni audio inputs require sound_context_token_id."
        case .imageContextTokenMissing:
            "Nemotron Omni image and video inputs require img_context_token_id."
        case .audioMetadataMissing:
            "Nemotron Omni audio metadata is missing; use NemotronOmniProcessor for audio inputs."
        case .soundFeatureCountMismatch(let expected, let actual):
            "Nemotron Omni sound features (\(expected)) do not match sound token count (\(actual))."
        case .visionFeatureCountMismatch(let expected, let actual):
            "Nemotron Omni vision features (\(expected)) do not match image token count (\(actual))."
        case .videoPruningUnsupported:
            "Nemotron Omni video pruning is not implemented."
        }
    }
}

public final class NemotronOmni: Module, VLMModel, KVCacheDimensionProvider,
    SeparateExpertWeightsProviding
{
    public var reasoningConfig: ReasoningConfig? {
        config.modelType == "nemotron_h_omni" ? .thinkTagsWithEnableThinking : nil
    }

    public var toolCallFormat: ToolCallFormat? { .xmlFunction }

    public let config: NemotronOmniConfiguration
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") private var languageModel: NemotronOmniLanguageModel
    @ModuleInfo(key: "vision_model") private var visionModel: NemotronOmniVision.VisionModel
    @ModuleInfo(key: "mlp1") private var visionProjection: NemotronOmniVision.VisionProjection
    @ModuleInfo(key: "vision_final_layernorm") private var visionFinalLayerNorm: LayerNorm?
    @ModuleInfo(key: "sound_encoder") private var soundEncoder: NemotronOmniAudio.SoundEncoder?
    @ModuleInfo(key: "sound_projection") private var soundProjection:
        NemotronOmniAudio.SoundProjection?

    public var loraLayers: [Module] {
        languageModel.loraLayers
    }

    public init(_ config: NemotronOmniConfiguration) {
        guard let languageConfiguration = config.languageConfiguration else {
            preconditionFailure(NemotronOmniError.languageConfigurationMissing.localizedDescription)
        }
        self.config = config
        vocabularySize = languageConfiguration.vocabSize
        kvHeads = Array(languageConfiguration.hybridOverridePattern).compactMap { char in
            switch NemotronOmniHBlockType(from: char) {
            case .attention:
                languageConfiguration.numKeyValueHeads
            case .mamba:
                0
            case .mlp, .moe:
                nil
            }
        }
        _languageModel.wrappedValue = NemotronOmniLanguageModel(languageConfiguration)
        _visionModel.wrappedValue = NemotronOmniVision.VisionModel(config.visionConfiguration)
        _visionProjection.wrappedValue = NemotronOmniVision.VisionProjection(config)
        if config.visionFinalLayerNorm {
            _visionFinalLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.vitHiddenSize, eps: config.visionConfiguration.layerNormEpsilon)
        }
        if let soundConfiguration = config.soundConfiguration {
            _soundEncoder.wrappedValue = NemotronOmniAudio.SoundEncoder(soundConfiguration)
            _soundProjection.wrappedValue = NemotronOmniAudio.SoundProjection(
                soundConfiguration,
                llmHiddenSize: languageConfiguration.hiddenSize,
            )
        }
        super.init()
    }

    public func newCache(configuration: KVCacheConfiguration?) -> [KVCache] {
        languageModel.newCache(configuration: configuration)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let tokens =
            input.text.tokens.ndim == 1
            ? input.text.tokens.expandedDimensions(axis: 0) : input.text.tokens
        let inputEmbeddings = try getInputEmbeddings(
            inputIDs: tokens, image: input.image, video: input.video, audio: input.audio)
        let count = inputEmbeddings.dim(1)
        guard count > 0 else { throw VLMError.processing("Nemotron Omni received an empty prompt") }
        let step = prefill.chunkLength(forChunking: count) ?? count
        let errors = InferenceErrorCheck(self)
        var logits: MLXArray?
        try withPreparedCache(cache, lengths: input.text.sequenceLengths) {
            for start in stride(from: 0, to: count, by: max(1, step)) {
                try Task.checkCancellation()
                let range = start ..< min(start + step, count)
                logits = try languageModel(
                    nil, cache: cache, inputEmbeddings: inputEmbeddings[0..., range, 0...])
                eval(logits!, cache)
                try errors.check()
                prefill.progress?(range.upperBound, count)
            }
        }
        return .logits(.init(logits: logits!))
    }

    public func getInputEmbeddings(inputIDs: MLXArray, audio: LMInput.ProcessedAudio?) throws
        -> MLXArray
    {
        try getInputEmbeddings(inputIDs: inputIDs, image: nil, video: nil, audio: audio)
    }

    public func getInputEmbeddings(
        inputIDs: MLXArray,
        image: LMInput.ProcessedImage?,
        video: LMInput.ProcessedVideo?,
        audio: LMInput.ProcessedAudio?,
    )
        throws -> MLXArray
    {
        var inputEmbeddings = languageModel.inputEmbeddings(inputIDs)

        var visionPieces: [MLXArray] = []
        if let image {
            try visionPieces.append(extractImageFeatures(image))
        }
        if let video {
            try visionPieces.append(extractVideoFeatures(video))
        }
        if !visionPieces.isEmpty {
            let visionFeatures = concatenated(visionPieces, axis: 0)
            inputEmbeddings = try mergeFeatures(
                inputEmbeddings: inputEmbeddings,
                inputIDs: inputIDs,
                tokenID: requiredImageContextTokenID(),
                features: visionFeatures,
                featureName: "vision",
            )
        }

        guard let audio else {
            return inputEmbeddings
        }
        guard let soundEncoder, let soundProjection else {
            throw NemotronOmniError.soundConfigurationMissing
        }
        guard let soundContextTokenID = config.soundContextTokenID else {
            throw NemotronOmniError.soundContextTokenMissing
        }

        let metadata = try unpackAudioMetadata(audio)
        let encoded = soundEncoder(
            audio.features.asType(inputEmbeddings.dtype),
            attentionMask: metadata.attentionMask,
        )
        var soundFeatures = soundProjection(encoded).asType(inputEmbeddings.dtype)
        let outputLengths = NemotronOmniAudio.subsamplingOutputLength(
            metadata.featureLengths,
            config: soundEncoder.config,
        ).asArray(Int.self)

        var pieces: [MLXArray] = []
        pieces.reserveCapacity(outputLengths.count)
        for index in 0 ..< outputLengths.count {
            let length = min(outputLengths[index], soundFeatures.dim(1))
            if length > 0 {
                pieces.append(soundFeatures[index, ..<length, 0...])
            }
        }
        soundFeatures =
            pieces.isEmpty
            ? MLXArray.zeros([0, inputEmbeddings.dim(-1)], dtype: inputEmbeddings.dtype)
            : concatenated(pieces, axis: 0)

        if !metadata.tokenCounts.isEmpty {
            let expected = metadata.tokenCounts.reduce(0, +)
            if expected != soundFeatures.dim(0) {
                throw NemotronOmniError.soundFeatureCountMismatch(
                    expected: expected,
                    actual: soundFeatures.dim(0),
                )
            }
        }

        return try mergeFeatures(
            inputEmbeddings: inputEmbeddings,
            inputIDs: inputIDs,
            tokenID: soundContextTokenID,
            features: soundFeatures,
            featureName: "sound",
        )
    }

    private func requiredImageContextTokenID() throws -> Int {
        guard let imageContextTokenID = config.imageContextTokenID else {
            throw NemotronOmniError.imageContextTokenMissing
        }
        return imageContextTokenID
    }

    private func extractImageFeatures(_ image: LMInput.ProcessedImage) throws -> MLXArray {
        let frames =
            image.frames
            ?? Array(
                repeating: THW(1, image.pixels.dim(-2), image.pixels.dim(-1)),
                count: image.pixels.dim(0),
            )
        var pieces: [MLXArray] = []
        pieces.reserveCapacity(frames.count)
        for index in 0 ..< min(frames.count, image.pixels.dim(0)) {
            let frame = frames[index]
            let pixelValues = image.pixels[
                index ..< (index + 1),
                0...,
                ..<min(frame.h, image.pixels.dim(-2)),
                ..<min(frame.w, image.pixels.dim(-1)),
            ]
            let output = visionModel(
                pixelValues
                    .asType(visionModel.radioModel.model.patchGenerator.embedder.weight.dtype))
            pieces.append(projectVisionFeatures(output.features, frame: frame))
        }
        return pieces.isEmpty
            ? MLXArray.zeros([0, config.textConfiguration.hiddenSize], dtype: image.pixels.dtype)
            : concatenated(pieces, axis: 0)
    }

    private func extractVideoFeatures(_ video: LMInput.ProcessedVideo) throws -> MLXArray {
        guard config.videoPruningRate == 0 else {
            throw NemotronOmniError.videoPruningUnsupported
        }
        let frames =
            video.frames
            ?? Array(
                repeating: THW(1, video.pixels.dim(-2), video.pixels.dim(-1)),
                count: video.pixels.dim(0),
            )
        let temporal = max(config.videoTemporalPatchSize, 1)
        var pieces: [MLXArray] = []
        var index = 0
        while index < frames.count, index < video.pixels.dim(0) {
            let end = min(index + temporal, frames.count, video.pixels.dim(0))
            let groupFrames = Array(frames[index ..< end])
            let height = groupFrames.map(\.h).min() ?? video.pixels.dim(-2)
            let width = groupFrames.map(\.w).min() ?? video.pixels.dim(-1)
            var tubeletFrames: [MLXArray] = []
            tubeletFrames.reserveCapacity(temporal)
            for frameIndex in index ..< end {
                tubeletFrames.append(
                    video.pixels[
                        frameIndex ..< (frameIndex + 1),
                        0...,
                        ..<min(height, video.pixels.dim(-2)),
                        ..<min(width, video.pixels.dim(-1)),
                    ])
            }
            while tubeletFrames.count < temporal, let last = tubeletFrames.last {
                tubeletFrames.append(last)
            }
            let tubelet = concatenated(tubeletFrames, axis: 1)
            let output = visionModel(
                tubelet.asType(
                    visionModel.radioModel.model.patchGenerator.videoEmbedder.weight.dtype),
                useVideoEmbedder: true,
            )
            pieces.append(projectVisionFeatures(output.features, frame: THW(1, height, width)))
            index += temporal
        }
        return pieces.isEmpty
            ? MLXArray.zeros([0, config.textConfiguration.hiddenSize], dtype: video.pixels.dtype)
            : concatenated(pieces, axis: 0)
    }

    private func projectVisionFeatures(_ features: MLXArray, frame: THW) -> MLXArray {
        let features = visionFinalLayerNorm?(features) ?? features
        let patchRows = max(frame.h / config.visionConfiguration.patchSize, 1)
        let patchColumns = max(frame.w / config.visionConfiguration.patchSize, 1)
        var hidden = features.reshaped(features.dim(0), patchRows, patchColumns, -1)
        hidden = pixelShuffle(hidden, factor: config.visionConfiguration.downsampleFactor)
        hidden = hidden.reshaped(hidden.dim(0), -1, hidden.dim(-1))
        return visionProjection(hidden).reshaped(-1, config.textConfiguration.hiddenSize)
    }

    private func pixelShuffle(_ x: MLXArray, factor: Int) -> MLXArray {
        guard factor > 1 else {
            return x
        }
        let batch = x.dim(0)
        let height = x.dim(1)
        let width = x.dim(2)
        let channels = x.dim(3)
        let outputHeight = max(height / factor, 1)
        let outputWidth = max(width / factor, 1)
        return
            x
            .reshaped(batch, outputHeight, factor, outputWidth, factor, channels)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(batch, outputHeight, outputWidth, channels * factor * factor)
    }

    private func mergeFeatures(
        inputEmbeddings: MLXArray,
        inputIDs: MLXArray,
        tokenID: Int,
        features: MLXArray,
        featureName: String,
    )
        throws -> MLXArray
    {
        let tokenMask = inputIDs .== tokenID
        let tokenCount = tokenMask.asType(.int32).sum().item(Int.self)
        guard tokenCount == features.dim(0) else {
            if featureName == "vision" {
                throw NemotronOmniError.visionFeatureCountMismatch(
                    expected: features.dim(0),
                    actual: tokenCount,
                )
            }
            throw NemotronOmniError.soundFeatureCountMismatch(
                expected: features.dim(0),
                actual: tokenCount,
            )
        }
        guard features.dim(0) > 0 else {
            return inputEmbeddings
        }
        let expandedMask = broadcast(
            expandedDimensions(tokenMask, axis: -1), to: inputEmbeddings.shape)
        return nemotronOmniMaskedScatter(
            inputTensor: inputEmbeddings,
            mask: expandedMask,
            source: features.asType(inputEmbeddings.dtype),
        )
    }

    private func unpackAudioMetadata(_ audio: LMInput.ProcessedAudio) throws -> (
        tokenCounts: [Int],
        featureLengths: MLXArray,
        attentionMask: MLXArray?,
    ) {
        guard let metadata = audio.mask,
            metadata.ndim == 2,
            metadata.dim(0) == audio.features.dim(0),
            metadata.dim(1) >= 2
        else {
            throw NemotronOmniError.audioMetadataMissing
        }

        let rawValues = metadata.asArray(Int32.self)
        let rowWidth = metadata.dim(1)
        let batchSize = metadata.dim(0)
        var tokenCounts: [Int] = []
        var featureLengths: [Int32] = []
        var maskValues: [Int32] = []
        tokenCounts.reserveCapacity(batchSize)
        featureLengths.reserveCapacity(batchSize)

        for row in 0 ..< batchSize {
            let start = row * rowWidth
            tokenCounts.append(Int(rawValues[start]))
            featureLengths.append(rawValues[start + 1])
            maskValues.append(contentsOf: rawValues[(start + 2) ..< (start + rowWidth)])
        }

        let attentionMask =
            rowWidth > 2
            ? MLXArray(maskValues, [batchSize, rowWidth - 2])
            : nil as MLXArray?
        return (
            tokenCounts,
            MLXArray(featureLengths, [batchSize]),
            attentionMask,
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        sanitize(weights: weights, audioIsMLX: false)
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        sanitize(weights: weights, audioIsMLX: metadata["format"] == "mlx")
    }

    private func sanitize(weights: [String: MLXArray], audioIsMLX: Bool) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        var languageWeights: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (key, value) in NemotronOmniAudio.sanitizeWeights(weights, isMLXLayout: audioIsMLX) {
            if key == "vision_model.summary_idxs" || key == "vision_model.radio_model.summary_idxs"
            {
                continue
            }
            if key.hasPrefix("language_model.") {
                languageWeights[String(key.dropFirst("language_model.".count))] = value
            } else if key.hasPrefix("vision_projector.vision_final_layernorm.") {
                sanitized[String(key.dropFirst("vision_projector.".count))] = value
            } else if key.hasPrefix("mlp1.") {
                sanitized[Self.remapVisionProjectionKey(key)] = value
            } else {
                sanitized[key] = value
            }
        }

        for (key, value) in languageModel.sanitizeLanguageWeights(languageWeights) {
            sanitized["language_model.\(key)"] = value
        }
        return sanitized
    }

    func separateExpertWeightNames(for path: String, expert: Int) -> [String] {
        for (target, source) in [("fc1", "up_proj"), ("fc2", "down_proj")] {
            let marker = ".switch_mlp.\(target)."
            if path.contains(marker) {
                return [
                    path.replacingOccurrences(
                        of: marker, with: ".experts.\(expert).\(source).")
                ]
            }
        }
        return []
    }

    public var castPredicate: ((String) -> Bool)? {
        { key in
            !key.contains("e_score_correction_bias") && !key.contains("A_log")
        }
    }

    private static func remapVisionProjectionKey(_ key: String) -> String {
        var key = key
        for index in [0, 1, 3] {
            key = key.replacingOccurrences(
                of: "mlp1.\(index).",
                with: "mlp1.layers.\(index).",
            )
        }
        return key
    }
}

private func nemotronOmniMaskedScatter(
    inputTensor: MLXArray,
    mask: MLXArray,
    source: MLXArray,
)
    -> MLXArray
{
    let mask = mask.asType(.bool)
    let inputShape = mask.shape
    let resultFlat = broadcast(inputTensor, to: inputShape).flattened()
    let maskFlat = mask.flattened()
    let sourceFlat = source.flattened()
    let selection = cumsum(maskFlat.asType(.int32), axis: 0) - 1
    let bounded = selection % sourceFlat.dim(0)
    let selected = sourceFlat.take(bounded, axis: 0)
    return MLX.where(maskFlat, selected, resultFlat).reshaped(inputShape)
}

// MARK: - Audio

public enum NemotronOmniAudio {
    public static func sanitizeWeights(_ weights: [String: MLXArray], isMLXLayout: Bool = false)
        -> [String: MLXArray]
    {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (key, value) in weights {
            if key.hasPrefix("sound_encoder.encoder.feature_extractor.") {
                continue
            }
            if key.hasSuffix(".num_batches_tracked") {
                continue
            }

            let newKey = remapSubsamplingLayerKey(key)
            var adjusted = value
            if !isMLXLayout, newKey.hasPrefix("sound_encoder.encoder."), newKey.hasSuffix(".weight")
            {
                if adjusted.ndim == 3 {
                    adjusted = adjusted.transposed(0, 2, 1)
                } else if adjusted.ndim == 4 {
                    adjusted = adjusted.transposed(0, 2, 3, 1)
                }
            }
            sanitized[newKey] = adjusted
        }

        return sanitized
    }

    private static func remapSubsamplingLayerKey(_ key: String) -> String {
        let prefix = "sound_encoder.encoder.subsampling.layers."
        let remappedPrefix = "sound_encoder.encoder.subsampling."
        guard key.hasPrefix(prefix) else {
            return key
        }

        let suffix = String(key.dropFirst(prefix.count))
        let parts = suffix.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
        guard parts.count == 2, let layerIndex = Int(parts[0]) else {
            return key
        }

        let parameter = parts[1]
        // Native MLX keys already name depthwise_conv / pointwise_conv explicitly.
        guard parameter == "weight" || parameter == "bias" else { return key }
        switch layerIndex {
        case 0:
            return "\(remappedPrefix)conv0.\(parameter)"
        default:
            let shifted = layerIndex - 2
            guard shifted >= 0 else {
                return key
            }
            let pair = shifted / 3
            switch shifted % 3 {
            case 0:
                return "\(prefix)\(pair).depthwise_conv.\(parameter)"
            case 1:
                return "\(prefix)\(pair).pointwise_conv.\(parameter)"
            default:
                return key
            }
        }
    }

    public struct SoundFeatureExtractor: Sendable {
        public let config: NemotronOmniConfiguration.SoundConfiguration

        private let hopLength: Int
        private let fftLength: Int
        private let window: MaterializedArray
        private let melFilters: MaterializedArray

        public init(_ config: NemotronOmniConfiguration.SoundConfiguration) {
            self.config = config
            hopLength = max(config.hopLength, 1)
            fftLength = max(config.fftLength, 1)
            let window = Self.hanningWindow(
                windowLength: max(config.windowLength, 1),
                fftLength: max(config.fftLength, 1),
            )
            let melFilters = Self.makeSlaneyMelFilterBank(
                frequencyBins: max(config.fftLength, 1) / 2 + 1,
                melFilters: config.numMelBins,
                samplingRate: config.samplingRate,
            )
            let tables = materialize([window, melFilters])
            self.window = tables[0]
            self.melFilters = tables[1]
        }

        public func callAsFunction(_ rawAudio: [MLXArray]) throws -> (
            inputFeatures: MLXArray,
            attentionMask: MLXArray,
            fullLengths: MLXArray,
        ) {
            guard !rawAudio.isEmpty else {
                throw VLMError.processing("No audio samples were provided.")
            }

            var features: [MLXArray] = []
            var fullLengths: [Int] = []
            var validLengths: [Int] = []
            features.reserveCapacity(rawAudio.count)
            fullLengths.reserveCapacity(rawAudio.count)
            validLengths.reserveCapacity(rawAudio.count)

            for audio in rawAudio {
                let waveform = Gemma4AudioFeatureExtractor.monoFloatSamples(audio)
                guard waveform.dim(0) > 0 else {
                    throw VLMError.processing("Audio input is empty.")
                }

                var mel = logMelSpectrogram(waveform)
                let fullLength = mel.dim(0)
                let validLength = min(waveform.dim(0) / hopLength, fullLength)
                mel = normalize(mel, validLength: validLength)

                features.append(mel)
                fullLengths.append(fullLength)
                validLengths.append(validLength)
            }

            let maxLength = fullLengths.max() ?? 0
            let paddedFeatures = features.enumerated().map { index, feature in
                let pad = max(0, maxLength - fullLengths[index])
                guard pad > 0 else {
                    return feature
                }
                return padded(feature, widths: [.init((0, pad)), 0])
            }
            let masks = validLengths.map { validLength in
                MLXArray(
                    (0 ..< maxLength).map { index in Int32(index < validLength ? 1 : 0) },
                    [maxLength],
                )
            }

            return (
                stacked(paddedFeatures, axis: 0),
                stacked(masks, axis: 0),
                MLXArray(fullLengths.map(Int32.init), [fullLengths.count]),
            )
        }

        private func logMelSpectrogram(_ waveform: MLXArray) -> MLXArray {
            var waveform = waveform
            if let preemphasis = config.preemphasis, preemphasis != 0, waveform.dim(0) > 1 {
                waveform = concatenated(
                    [
                        waveform[..<1],
                        waveform[1...] - MLXArray(preemphasis, dtype: waveform.dtype)
                            * waveform[..<(waveform.dim(0) - 1)],
                    ],
                    axis: 0,
                )
            }

            let centerPad = fftLength / 2
            let paddedWaveform = padded(
                waveform,
                widths: [.init((centerPad, centerPad))],
            )
            let available = max(0, paddedWaveform.dim(0) - fftLength)
            let frameCount = max(1, available / hopLength + 1)

            var rows: [MLXArray] = []
            rows.reserveCapacity(frameCount)
            for frameIndex in 0 ..< frameCount {
                let start = frameIndex * hopLength
                let availableSamples = max(0, min(fftLength, paddedWaveform.dim(0) - start))
                var frame =
                    availableSamples > 0
                    ? paddedWaveform[start ..< (start + availableSamples)]
                    : MLXArray.zeros([0], dtype: .float32)
                if availableSamples < fftLength {
                    frame = padded(frame, widths: [.init((0, fftLength - availableSamples))])
                }
                rows.append(frame)
            }

            let frames = stacked(rows, axis: 0) * window.asType(waveform.dtype)
            let spectrum = MLXFFT.rfft(frames, n: fftLength, axis: -1)
            let power = spectrum.abs().square()
            let mel = power.matmul(melFilters.asType(power.dtype))
            return log(mel + MLXArray(Float(pow(2.0, -24.0)), dtype: mel.dtype))
                .asType(.float32)
        }

        private func normalize(_ mel: MLXArray, validLength: Int) -> MLXArray {
            let mask = MLXArray(
                (0 ..< mel.dim(0)).map { index in Float(index < validLength ? 1 : 0) },
                [mel.dim(0), 1],
            )
            let maskForDType = mask.asType(mel.dtype)
            let denom = MLXArray(Float(max(validLength, 1)), dtype: mel.dtype)
            let varianceDenom = MLXArray(Float(max(validLength - 1, 1)), dtype: mel.dtype)
            let mean = sum(mel * maskForDType, axis: 0) / denom
            let variance = sum(((mel - mean).square()) * maskForDType, axis: 0) / varianceDenom
            return ((mel - mean) / (sqrt(variance) + MLXArray(1e-5, dtype: mel.dtype)))
                * maskForDType
        }

        private static func hanningWindow(windowLength: Int, fftLength: Int) -> MLXArray {
            let values: [Float] =
                if windowLength == 1 {
                    [1]
                } else {
                    (0 ..< windowLength).map { index in
                        Float(
                            0.5 - 0.5
                                * cos(
                                    2.0 * Double.pi * Double(index) / Double(windowLength - 1),
                                ),
                        )
                    }
                }

            if windowLength == fftLength {
                return MLXArray(values, [fftLength])
            }
            if windowLength < fftLength {
                let left = (fftLength - windowLength) / 2
                let right = fftLength - windowLength - left
                return MLXArray(
                    Array(repeating: Float(0), count: left)
                        + values
                        + Array(repeating: Float(0), count: right),
                    [fftLength],
                )
            }
            return MLXArray(Array(values.prefix(fftLength)), [fftLength])
        }

        private static func hertzToMel(_ frequency: Float) -> Float {
            let minLogHertz: Float = 1000
            let minLogMel: Float = 15
            let logStep = log(Float(6.4)) / 27
            if frequency < minLogHertz {
                return frequency / (200 / 3)
            }
            return minLogMel + log(frequency / minLogHertz) / logStep
        }

        private static func melToHertz(_ mel: Float) -> Float {
            let minLogHertz: Float = 1000
            let minLogMel: Float = 15
            let logStep = log(Float(6.4)) / 27
            if mel < minLogMel {
                return mel * (200 / 3)
            }
            return minLogHertz * exp(logStep * (mel - minLogMel))
        }

        private static func makeSlaneyMelFilterBank(
            frequencyBins: Int,
            melFilters: Int,
            samplingRate: Int,
        )
            -> MLXArray
        {
            let melMin = hertzToMel(0)
            let melMax = hertzToMel(Float(samplingRate) / 2)
            let melPoints = (0 ..< melFilters + 2).map { index in
                melMin + (melMax - melMin) * Float(index) / Float(melFilters + 1)
            }
            let frequencyPoints = melPoints.map(melToHertz)
            let fftFrequencies = (0 ..< frequencyBins).map { index in
                Float(index) * Float(samplingRate) / Float(2 * max(frequencyBins - 1, 1))
            }

            var values = Array(repeating: Float(0), count: frequencyBins * melFilters)
            for filterIndex in 0 ..< melFilters {
                let lower = frequencyPoints[filterIndex]
                let center = frequencyPoints[filterIndex + 1]
                let upper = frequencyPoints[filterIndex + 2]
                let enorm = 2.0 / max(upper - lower, 1e-10)
                for binIndex in 0 ..< frequencyBins {
                    let frequency = fftFrequencies[binIndex]
                    let rising = (frequency - lower) / max(center - lower, 1e-10)
                    let falling = (upper - frequency) / max(upper - center, 1e-10)
                    values[binIndex * melFilters + filterIndex] =
                        max(0, min(rising, falling)) * enorm
                }
            }
            return MLXArray(values, [frequencyBins, melFilters])
        }
    }

    public final class SoundProjection: Module, UnaryLayer {
        @ModuleInfo var norm: RMSNorm
        @ModuleInfo var linear1: Linear
        @ModuleInfo var linear2: Linear

        public init(_ config: NemotronOmniConfiguration.SoundConfiguration, llmHiddenSize: Int) {
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-5)
            _linear1.wrappedValue = Linear(
                config.hiddenSize,
                config.projectionHiddenSize,
                bias: config.projectionBias,
            )
            _linear2.wrappedValue = Linear(
                config.projectionHiddenSize,
                llmHiddenSize,
                bias: config.projectionBias,
            )
            super.init()
        }

        public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            var hiddenStates = norm(hiddenStates)
            hiddenStates = linear1(hiddenStates)
            hiddenStates = relu(hiddenStates).square()
            return linear2(hiddenStates)
        }
    }

    public final class SoundEncoder: Module {
        public let config: NemotronOmniConfiguration.SoundConfiguration

        @ModuleInfo(key: "encoder") var encoder: ParakeetEncoder

        public init(_ config: NemotronOmniConfiguration.SoundConfiguration) {
            self.config = config
            _encoder.wrappedValue = ParakeetEncoder(config)
            super.init()
        }

        public var hiddenSize: Int {
            config.hiddenSize
        }

        public func callAsFunction(_ inputFeatures: MLXArray, attentionMask: MLXArray? = nil)
            -> MLXArray
        {
            let (hiddenStates, _) = encoder(inputFeatures, attentionMask: attentionMask)
            return hiddenStates
        }

        public func encoded(_ inputFeatures: MLXArray, attentionMask: MLXArray? = nil)
            -> (MLXArray, MLXArray?)
        {
            encoder(inputFeatures, attentionMask: attentionMask)
        }
    }

    public static func subsamplingOutputLength(
        _ inputLengths: MLXArray,
        config: NemotronOmniConfiguration.SoundConfiguration,
    )
        -> MLXArray
    {
        var lengths = inputLengths
        let kernelSize = config.subsamplingConvKernelSize
        let stride = config.subsamplingConvStride
        let padding = (kernelSize - 1) / 2
        let addPad = 2 * padding - kernelSize
        for _ in 0 ..< config.subsamplingLayerCount {
            lengths = floor((lengths.asType(.float32) + Float(addPad)) / Float(stride) + 1.0)
        }
        return lengths.asType(.int32)
    }

    final class ParakeetEncoder: Module {
        let config: NemotronOmniConfiguration.SoundConfiguration
        let inputScale: Float

        @ModuleInfo var subsampling: SubsamplingConv2D
        @ModuleInfo(key: "encode_positions") var encodePositions: RelPositionalEncoding
        @ModuleInfo var layers: [ParakeetEncoderBlock]

        init(_ config: NemotronOmniConfiguration.SoundConfiguration) {
            self.config = config
            inputScale = config.scaleInput ? sqrt(Float(config.hiddenSize)) : 1.0
            _subsampling.wrappedValue = SubsamplingConv2D(config)
            _encodePositions.wrappedValue = RelPositionalEncoding(config)
            _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { layerIndex in
                ParakeetEncoderBlock(config, layerIndex: layerIndex)
            }
            super.init()
        }

        func callAsFunction(_ inputFeatures: MLXArray, attentionMask: MLXArray? = nil)
            -> (MLXArray, MLXArray?)
        {
            var hiddenStates = subsampling(inputFeatures, attentionMask: attentionMask)
            hiddenStates = hiddenStates * inputScale
            let positionEmbeddings = encodePositions(hiddenStates)

            var blockAttentionMask: MLXArray?
            var outputMask: MLXArray?
            if let attentionMask {
                outputMask = outputAttentionMask(attentionMask, targetLength: hiddenStates.dim(1))
                if let outputMask {
                    blockAttentionMask = logicalAnd(
                        outputMask.expandedDimensions(axis: 1).expandedDimensions(axis: 3),
                        outputMask.expandedDimensions(axis: 1).expandedDimensions(axis: 2),
                    )
                }
            }

            for layer in layers {
                hiddenStates = layer(
                    hiddenStates,
                    attentionMask: blockAttentionMask,
                    positionEmbeddings: positionEmbeddings,
                )
            }

            return (hiddenStates, outputMask)
        }

        private func outputAttentionMask(_ attentionMask: MLXArray, targetLength: Int) -> MLXArray {
            let inputLengths = sum(attentionMask, axis: -1).asType(.int32)
            let outputLengths = subsamplingOutputLength(inputLengths, config: config)
            return arange(0, targetLength, dtype: .int32)
                .expandedDimensions(axis: 0) .< outputLengths.expandedDimensions(axis: 1)
        }
    }

    final class SubsamplingConv2D: Module {
        let config: NemotronOmniConfiguration.SoundConfiguration
        let kernelSize: Int
        let stride: Int
        let padding: Int

        @ModuleInfo var conv0: Conv2d
        @ModuleInfo var layers: [DepthwisePointwiseConv2D]
        @ModuleInfo var linear: Linear

        init(_ config: NemotronOmniConfiguration.SoundConfiguration) {
            self.config = config
            kernelSize = config.subsamplingConvKernelSize
            stride = config.subsamplingConvStride
            padding = (kernelSize - 1) / 2
            _conv0.wrappedValue = Conv2d(
                inputChannels: 1,
                outputChannels: config.subsamplingConvChannels,
                kernelSize: .init(kernelSize),
                stride: .init(stride),
                padding: .init(padding),
            )
            _layers.wrappedValue = (0 ..< max(config.subsamplingLayerCount - 1, 0)).map { _ in
                DepthwisePointwiseConv2D(config)
            }
            let outputFrequency = Self.outputFrequencyLength(config: config)
            _linear.wrappedValue = Linear(
                config.subsamplingConvChannels * outputFrequency,
                config.hiddenSize,
                bias: true,
            )
            super.init()
        }

        func callAsFunction(_ inputFeatures: MLXArray, attentionMask: MLXArray? = nil)
            -> MLXArray
        {
            var hiddenStates = inputFeatures.expandedDimensions(axis: -1)
            var currentLengths =
                attentionMask.map { sum($0, axis: -1).asType(.int32) }

            hiddenStates = relu(conv0(hiddenStates))
            currentLengths = updateAndApplyMask(
                currentLengths,
                hiddenStates: &hiddenStates,
            )

            for layer in layers {
                hiddenStates = layer(hiddenStates)
                currentLengths = updateAndApplyMask(
                    currentLengths,
                    hiddenStates: &hiddenStates,
                )
            }

            let batchSize = hiddenStates.dim(0)
            let sequenceLength = hiddenStates.dim(1)
            let frequency = hiddenStates.dim(2)
            let channels = hiddenStates.dim(3)
            hiddenStates =
                hiddenStates
                .transposed(0, 1, 3, 2)
                .reshaped(batchSize, sequenceLength, channels * frequency)
            return linear(hiddenStates)
        }

        private func updateAndApplyMask(_ lengths: MLXArray?, hiddenStates: inout MLXArray)
            -> MLXArray?
        {
            guard let lengths else {
                return nil
            }
            let updatedLengths = outputLength(lengths)
            let currentLength = hiddenStates.dim(1)
            let mask =
                arange(0, currentLength, dtype: .int32)
                .expandedDimensions(axis: 0) .< updatedLengths.expandedDimensions(axis: 1)
            hiddenStates =
                hiddenStates
                * mask.expandedDimensions(axis: -1).expandedDimensions(axis: -1)
                .asType(hiddenStates.dtype)
            return updatedLengths
        }

        private func outputLength(_ inputLengths: MLXArray) -> MLXArray {
            floor(
                (inputLengths.asType(.float32) + Float(2 * padding - kernelSize))
                    / Float(stride) + 1.0,
            )
            .asType(.int32)
        }

        private static func outputFrequencyLength(
            config: NemotronOmniConfiguration.SoundConfiguration,
        )
            -> Int
        {
            var length = config.numMelBins
            let kernelSize = config.subsamplingConvKernelSize
            let stride = config.subsamplingConvStride
            let padding = (kernelSize - 1) / 2
            for _ in 0 ..< config.subsamplingLayerCount {
                length = ((length + 2 * padding - kernelSize) / stride) + 1
            }
            return max(length, 1)
        }
    }

    final class DepthwisePointwiseConv2D: Module, UnaryLayer {
        @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv2d
        @ModuleInfo(key: "pointwise_conv") var pointwiseConv: Conv2d

        init(_ config: NemotronOmniConfiguration.SoundConfiguration) {
            _depthwiseConv.wrappedValue = Conv2d(
                inputChannels: config.subsamplingConvChannels,
                outputChannels: config.subsamplingConvChannels,
                kernelSize: .init(config.subsamplingConvKernelSize),
                stride: .init(config.subsamplingConvStride),
                padding: .init((config.subsamplingConvKernelSize - 1) / 2),
                groups: config.subsamplingConvChannels,
            )
            _pointwiseConv.wrappedValue = Conv2d(
                inputChannels: config.subsamplingConvChannels,
                outputChannels: config.subsamplingConvChannels,
                kernelSize: .init(1),
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            relu(pointwiseConv(depthwiseConv(x)))
        }
    }

    final class RelPositionalEncoding: Module {
        let hiddenSize: Int
        let maxPositionEmbeddings: Int

        init(_ config: NemotronOmniConfiguration.SoundConfiguration) {
            hiddenSize = config.hiddenSize
            maxPositionEmbeddings = config.maxPositionEmbeddings
            super.init()
        }

        func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            let sequenceLength = hiddenStates.dim(1)
            precondition(
                sequenceLength <= maxPositionEmbeddings,
                "Sequence length exceeds max_position_embeddings.",
            )

            let positions = arange(
                sequenceLength - 1,
                -sequenceLength,
                step: -1,
                dtype: .float32,
            )
            let inverseFrequency =
                1.0
                / pow(
                    MLXArray(10000.0, dtype: .float32),
                    arange(0, hiddenSize, step: 2, dtype: .float32) / Float(hiddenSize),
                )
            let frequencies =
                positions.expandedDimensions(axis: 1) * inverseFrequency.expandedDimensions(axis: 0)
            let positional = stacked([sin(frequencies), cos(frequencies)], axis: -1)
                .reshaped(2 * sequenceLength - 1, hiddenSize)
            return broadcast(
                positional.expandedDimensions(axis: 0),
                to: [hiddenStates.dim(0), positional.dim(0), positional.dim(1)],
            )
            .asType(hiddenStates.dtype)
        }
    }

    final class FeedForward: Module, UnaryLayer {
        @ModuleInfo var linear1: Linear
        @ModuleInfo var linear2: Linear

        init(_ config: NemotronOmniConfiguration.SoundConfiguration) {
            _linear1.wrappedValue = Linear(
                config.hiddenSize,
                config.intermediateSize,
                bias: config.attentionBias,
            )
            _linear2.wrappedValue = Linear(
                config.intermediateSize,
                config.hiddenSize,
                bias: config.attentionBias,
            )
            super.init()
        }

        func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            linear2(silu(linear1(hiddenStates)))
        }
    }

    final class ConvolutionModule: Module {
        @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
        @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
        @ModuleInfo var norm: BatchNorm
        @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d

        init(_ config: NemotronOmniConfiguration.SoundConfiguration) {
            let channels = config.hiddenSize
            _pointwiseConv1.wrappedValue = Conv1d(
                inputChannels: channels,
                outputChannels: 2 * channels,
                kernelSize: 1,
                bias: config.convolutionBias,
            )
            _depthwiseConv.wrappedValue = Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: config.convKernelSize,
                padding: (config.convKernelSize - 1) / 2,
                groups: channels,
                bias: config.convolutionBias,
            )
            _norm.wrappedValue = BatchNorm(featureCount: channels)
            _pointwiseConv2.wrappedValue = Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: 1,
                bias: config.convolutionBias,
            )
            super.init()
        }

        func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray? = nil)
            -> MLXArray
        {
            var hiddenStates = glu(pointwiseConv1(hiddenStates), axis: -1)
            if let attentionMask {
                let allMaskedRows = all(logicalNot(attentionMask), axis: -1)
                    .squeezed(axis: 1)
                hiddenStates = MLX.where(
                    allMaskedRows.expandedDimensions(axis: -1),
                    MLXArray(0.0, dtype: hiddenStates.dtype),
                    hiddenStates,
                )
            }
            hiddenStates = depthwiseConv(hiddenStates)
            hiddenStates = norm(hiddenStates)
            hiddenStates = silu(hiddenStates)
            return pointwiseConv2(hiddenStates)
        }
    }

    final class Attention: Module {
        let numHeads: Int
        let headDim: Int
        let scaling: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "relative_k_proj") var relativeKProj: Linear
        @ModuleInfo(key: "bias_u") var biasU: MLXArray
        @ModuleInfo(key: "bias_v") var biasV: MLXArray

        init(_ config: NemotronOmniConfiguration.SoundConfiguration, layerIndex _: Int) {
            numHeads = config.numAttentionHeads
            headDim = config.hiddenSize / config.numAttentionHeads
            scaling = pow(Float(headDim), -0.5)
            _qProj.wrappedValue = Linear(
                config.hiddenSize,
                config.numAttentionHeads * headDim,
                bias: config.attentionBias,
            )
            _kProj.wrappedValue = Linear(
                config.hiddenSize,
                config.numAttentionHeads * headDim,
                bias: config.attentionBias,
            )
            _vProj.wrappedValue = Linear(
                config.hiddenSize,
                config.numAttentionHeads * headDim,
                bias: config.attentionBias,
            )
            _oProj.wrappedValue = Linear(
                config.numAttentionHeads * headDim,
                config.hiddenSize,
                bias: config.attentionBias,
            )
            _relativeKProj.wrappedValue = Linear(
                config.hiddenSize,
                config.numAttentionHeads * headDim,
                bias: false,
            )
            _biasU.wrappedValue = MLXArray.zeros([config.numAttentionHeads, headDim])
            _biasV.wrappedValue = MLXArray.zeros([config.numAttentionHeads, headDim])
            super.init()
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            positionEmbeddings: MLXArray,
            attentionMask: MLXArray? = nil,
        )
            -> MLXArray
        {
            let batchSize = hiddenStates.dim(0)
            let sequenceLength = hiddenStates.dim(1)
            let hiddenShape = [batchSize, sequenceLength, numHeads, headDim]

            let queryStates = qProj(hiddenStates)
                .reshaped(hiddenShape)
                .transposed(0, 2, 1, 3)
            let keyStates = kProj(hiddenStates)
                .reshaped(hiddenShape)
                .transposed(0, 2, 1, 3)
            let valueStates = vProj(hiddenStates)
                .reshaped(hiddenShape)
                .transposed(0, 2, 1, 3)

            let queryStatesWithBiasU =
                queryStates + biasU.expandedDimensions(axis: 0).expandedDimensions(axis: 2)
            let queryStatesWithBiasV =
                queryStates + biasV.expandedDimensions(axis: 0).expandedDimensions(axis: 2)

            let relativeKeyStates = relativeKProj(positionEmbeddings)
                .reshaped(batchSize, -1, numHeads, headDim)

            var matrixBD = matmul(
                queryStatesWithBiasV,
                relativeKeyStates.transposed(0, 2, 3, 1),
            )
            matrixBD = relativeShift(matrixBD)[0..., 0..., 0..., ..<sequenceLength] * scaling

            if let attentionMask {
                matrixBD = MLX.where(
                    attentionMask,
                    matrixBD,
                    MLXArray(-1e9, dtype: matrixBD.dtype),
                )
            }

            var attentionOutput = MLXFast.scaledDotProductAttention(
                queries: queryStatesWithBiasU,
                keys: keyStates,
                values: valueStates,
                scale: scaling,
                mask: .array(matrixBD),
            )
            if let attentionMask {
                let validQueries = any(attentionMask, axis: -1)
                attentionOutput =
                    attentionOutput
                    * validQueries.expandedDimensions(axis: -1)
                    .asType(attentionOutput.dtype)
            }

            attentionOutput =
                attentionOutput
                .transposed(0, 2, 1, 3)
                .reshaped(batchSize, sequenceLength, -1)
            return oProj(attentionOutput)
        }

        private func relativeShift(_ attentionScores: MLXArray) -> MLXArray {
            let batchSize = attentionScores.dim(0)
            let heads = attentionScores.dim(1)
            let queryLength = attentionScores.dim(2)
            let positionLength = attentionScores.dim(3)
            var scores = padded(attentionScores, widths: [0, 0, 0, .init((1, 0))])
            scores = scores.reshaped(batchSize, heads, positionLength + 1, queryLength)
            scores = scores[0..., 0..., 1..., 0...]
            return scores.reshaped(batchSize, heads, queryLength, positionLength)
        }
    }

    final class ParakeetEncoderBlock: Module {
        @ModuleInfo(key: "feed_forward1") var feedForward1: FeedForward
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo var conv: ConvolutionModule
        @ModuleInfo(key: "feed_forward2") var feedForward2: FeedForward
        @ModuleInfo(key: "norm_feed_forward1") var normFeedForward1: LayerNorm
        @ModuleInfo(key: "norm_self_att") var normSelfAttention: LayerNorm
        @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
        @ModuleInfo(key: "norm_feed_forward2") var normFeedForward2: LayerNorm
        @ModuleInfo(key: "norm_out") var normOut: LayerNorm

        init(_ config: NemotronOmniConfiguration.SoundConfiguration, layerIndex: Int) {
            _feedForward1.wrappedValue = FeedForward(config)
            _selfAttention.wrappedValue = Attention(config, layerIndex: layerIndex)
            _conv.wrappedValue = ConvolutionModule(config)
            _feedForward2.wrappedValue = FeedForward(config)
            _normFeedForward1.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
            _normSelfAttention.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
            _normConv.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
            _normFeedForward2.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
            _normOut.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
            super.init()
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            attentionMask: MLXArray? = nil,
            positionEmbeddings: MLXArray,
        )
            -> MLXArray
        {
            var residual = hiddenStates
            var hiddenStates = feedForward1(normFeedForward1(hiddenStates))
            hiddenStates = residual + 0.5 * hiddenStates

            let normalizedHiddenStates = normSelfAttention(hiddenStates)
            hiddenStates =
                hiddenStates
                + selfAttention(
                    normalizedHiddenStates,
                    positionEmbeddings: positionEmbeddings,
                    attentionMask: attentionMask,
                )

            hiddenStates =
                hiddenStates + conv(normConv(hiddenStates), attentionMask: attentionMask)
            residual = hiddenStates
            hiddenStates = feedForward2(normFeedForward2(hiddenStates))
            hiddenStates = residual + 0.5 * hiddenStates
            return normOut(hiddenStates)
        }
    }
}
