import Foundation
import MLX
import MLXNN

class MambaMixer: Module {
    let stateSize: Int
    let convKernelSize: Int
    let intermediateSize: Int
    let timeStepRank: Int
    let useBCDTRMS: Bool
    let mixerRMSEps: Float

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "x_proj") var xProj: Linear
    @ModuleInfo(key: "dt_proj") var dtProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "D") var d: MLXArray

    init(_ config: MambaConfiguration) {
        self.stateSize = config.stateSize
        self.convKernelSize = config.convKernel
        self.intermediateSize = config.intermediateSize
        self.timeStepRank = config.resolvedTimeStepRank
        self.useBCDTRMS = config.useBCDTRMS
        self.mixerRMSEps = config.mixerRMSEps

        _inProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize * 2, bias: config.useBias)
        // Depthwise: one filter per channel, which is what `groups == channels` means.
        _conv1d.wrappedValue = Conv1d(
            inputChannels: config.intermediateSize,
            outputChannels: config.intermediateSize,
            kernelSize: config.convKernel,
            padding: 0,
            groups: config.intermediateSize,
            bias: config.useConvBias)
        _xProj.wrappedValue = Linear(
            config.intermediateSize, timeStepRank + 2 * config.stateSize, bias: false)
        _dtProj.wrappedValue = Linear(timeStepRank, config.intermediateSize, bias: true)
        _outProj.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: config.useBias)

        // Initialize stable state-space dynamics before loading checkpoint weights.
        let stateIndices = MLXArray(1 ... config.stateSize).asType(.float32)
            .reshaped(1, config.stateSize)
        _aLog.wrappedValue = MLX.log(
            MLX.repeated(stateIndices, count: config.intermediateSize, axis: 0))
        _d.wrappedValue = MLXArray.ones([config.intermediateSize])
    }

    /// One position through the recurrence, returning its output and the carried state.
    private func ssmStep(_ x: MLXArray, stateMatrix: MLXArray, state: MLXArray?)
        -> (MLXArray, MLXArray)
    {
        let projected = xProj(x)
        var delta = projected[0..., ..<timeStepRank]
        var b = projected[0..., timeStepRank ..< (timeStepRank + stateSize)]
        var c = projected[0..., (timeStepRank + stateSize)...]

        // falcon_mamba only: normalize the three projections before they are used.
        if useBCDTRMS {
            delta = MLXFast.rmsNorm(delta, weight: MLXArray.mlxNone, eps: mixerRMSEps)
            b = MLXFast.rmsNorm(b, weight: MLXArray.mlxNone, eps: mixerRMSEps)
            c = MLXFast.rmsNorm(c, weight: MLXArray.mlxNone, eps: mixerRMSEps)
        }

        delta = softplus(dtProj(delta))
        var newState = expandedDimensions(delta * x, axis: -1) * expandedDimensions(b, axis: 1)
        if let state {
            newState = newState + state * exp(expandedDimensions(delta, axis: -1) * stateMatrix)
        }
        let output = matmul(newState, expandedDimensions(c, axis: -1)).squeezed(axis: 2)
        return (output + d * x, newState)
    }

    func callAsFunction(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        let split = MLX.split(inProj(x), parts: 2, axis: -1)
        var hidden = split[0]
        let gate = split[1]

        // The convolution needs the previous window. Cold, that window is zeros; warm, it
        // is the tail the last call kept, which is what makes a one-token step equal to the
        // same position inside a longer prompt.
        let full: MLXArray
        if let previous = cache?[0] {
            full = concatenated([previous, hidden], axis: 1)
        } else {
            full = padded(
                hidden, widths: [.init((0, 0)), .init((convKernelSize - 1, 0)), .init((0, 0))])
        }
        let convolved = conv1d(full)
        let keptWindow = full[0..., (full.dim(1) - (convKernelSize - 1))..., 0...]
        hidden = silu(convolved)

        let stateMatrix = -exp(aLog)
        var state = cache?[1]
        var outputs: [MLXArray] = []
        for position in 0 ..< hidden.dim(1) {
            let (output, next) = ssmStep(
                hidden[0..., position], stateMatrix: stateMatrix, state: state)
            state = next
            outputs.append(output)
        }
        let stacked = MLX.stacked(outputs, axis: 1)

        if let cache {
            cache[0] = keptWindow
            cache[1] = state
        }

        return outProj(compiledSiluProduct(gate, stacked))
    }
}

class MambaResidualBlock: Module {
    let mixer: MambaMixer
    let norm: RMSNorm

    init(_ config: MambaConfiguration) {
        self.mixer = MambaMixer(config)
        self.norm = RMSNorm(dimensions: config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        mixer(norm(x), cache: cache) + x
    }
}

public class MambaBackbone: Module {
    @ModuleInfo(key: "embeddings") var embeddings: Embedding

    let layers: [MambaResidualBlock]

    @ModuleInfo(key: "norm_f") var normF: RMSNorm

    public init(_ config: MambaConfiguration) {
        precondition(config.vocabularySize > 0)
        _embeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { _ in MambaResidualBlock(config) }
        _normF.wrappedValue = RMSNorm(dimensions: config.hiddenSize)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hidden = embeddings(inputs)
        for (i, layer) in layers.enumerated() {
            hidden = layer(hidden, cache: cache?[i] as? ArraysCache)
        }
        return normF(hidden)
    }
}

public class MambaLanguageModel: Module {
    public let backbone: MambaBackbone

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: MambaConfiguration) {
        self.backbone = MambaBackbone(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = backbone(inputs, cache: cache)
        if let lmHead {
            return lmHead(hidden)
        }
        return backbone.embeddings.asLinear(hidden)
    }
}

public class MambaModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: MambaLanguageModel

    let configuration: MambaConfiguration

    public init(_ config: MambaConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        // No attention, so no key-value heads. The count still has to match the layer count
        // for the cache machinery that reads it.
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in 0 }
        _languageModel.wrappedValue = MambaLanguageModel(config)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    /// Two slots per layer: the convolution window and the recurrent state.
    ///
    /// Both are fixed size, so this cache never grows with the prompt -- and neither can it
    /// be trimmed, which is why a speculative loop over this model would have to snapshot
    /// rather than roll back.
    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        (0 ..< self.configuration.hiddenLayers).map { _ in ArraysCache(size: 2) }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            var name = key
            if name.hasPrefix("language_model.") {
                name = String(name.dropFirst("language_model.".count))
            }
            var array = value
            // A published depthwise convolution is `[channels, 1, kernel]`; this Conv1d
            // wants `[channels, kernel, 1]`. Both are three-dimensional and the same size,
            // so binding the wrong one convolves over a single tap and answers.
            if name.hasSuffix("conv1d.weight"), array.ndim == 3, array.dim(-1) != 1 {
                array = array.movedAxis(source: 2, destination: 1)
            }
            if configuration.tieWordEmbeddings, name.hasPrefix("lm_head.") {
                continue
            }
            sanitized["language_model." + name] = array
        }
        return sanitized
    }
}

public struct MambaConfiguration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var stateSize: Int
    var hiddenLayers: Int
    var convKernel: Int
    var useBias: Bool
    var useConvBias: Bool
    /// `nil` when the config said "auto", which resolves against `hidden_size`.
    var timeStepRank: Int?
    var tieWordEmbeddings: Bool
    var useBCDTRMS: Bool
    var mixerRMSEps: Float

    var resolvedTimeStepRank: Int {
        timeStepRank ?? Int((Double(hiddenSize) / 16).rounded(.up))
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case stateSize = "state_size"
        case hiddenLayers = "num_hidden_layers"
        case convKernel = "conv_kernel"
        case useBias = "use_bias"
        case useConvBias = "use_conv_bias"
        case timeStepRank = "time_step_rank"
        case tieWordEmbeddings = "tie_word_embeddings"
        case useBCDTRMS = "use_bcdt_rms"
        case mixerRMSEps = "mixer_rms_eps"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "mamba"
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        stateSize = try c.decode(Int.self, forKey: .stateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        convKernel = try c.decode(Int.self, forKey: .convKernel)
        useBias = try c.decode(Bool.self, forKey: .useBias)
        useConvBias = try c.decode(Bool.self, forKey: .useConvBias)

        // `time_step_rank` is a number or the string "auto". Decoding it as an Int only
        // would throw on half the published configs.
        if let rank = try? c.decode(Int.self, forKey: .timeStepRank) {
            timeStepRank = rank
        } else {
            let spelled = try c.decode(String.self, forKey: .timeStepRank)
            guard spelled == "auto" else {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: c.codingPath,
                        debugDescription: "time_step_rank must be a number or \"auto\", got "
                            + "\"\(spelled)\""))
            }
            timeStepRank = nil
        }

        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        mixerRMSEps = try c.decodeIfPresent(Float.self, forKey: .mixerRMSEps) ?? 1e-6
        // FalconMamba always normalizes B, C, and delta projections.
        useBCDTRMS =
            try modelType == "falcon_mamba"
            || (c.decodeIfPresent(Bool.self, forKey: .useBCDTRMS) ?? false)
    }
}

extension MambaModel: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.backbone.layers
    }
}
