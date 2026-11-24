import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct KimiK3Configuration: Decodable, Sendable {

    public struct TextConfiguration: Decodable, Sendable {
        public var modelType: String = "kimi_linear"
        public var vocabularySize: Int = 163_840
        public var hiddenSize: Int = 7168
        public var hiddenLayers: Int = 93
        public var attentionHeads: Int = 96
        public var kvHeads: Int = 96
        public var intermediateSize: Int = 33792
        public var rmsNormEps: Float = 1e-5
        public var situBeta: Float?
        public var situLinearBeta: Float?
        public var attentionResidualBlockSize: Int?
        public var qLoraRank: Int?
        public var kvLoraRank: Int?
        public var qkNopeHeadDim: Int?
        public var qkRopeHeadDim: Int?
        public var vHeadDim: Int?
        public var mlaUseNope: Bool = true
        public var mlaUseOutputGate: Bool = false
        public var experts: Int?
        public var expertsPerToken: Int = 16
        public var sharedExperts: Int = 0
        public var moeIntermediateSize: Int?
        public var moeRenormalize: Bool = true
        public var routedScalingFactor: Float = 1.0
        public var firstKDenseReplace: Int = 0
        public var moeLayerFrequency: Int = 1
        public var expertGroups: Int = 1
        public var topkGroups: Int = 1
        public var routedExpertHiddenSize: Int?
        public var latentMoEUseNorm: Bool = false
        public var tieWordEmbeddings: Bool = false

        public var linearHeads: Int = 0
        public var linearHeadDim: Int = 0
        public var linearConvKernel: Int = 4
        public var linearLowerBound: Float?
        public var linearUsesFullRankGate: Bool = false
        /// One-based, as the checkpoint writes it.
        public var kdaLayers: [Int] = []

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabularySize = "vocab_size"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case intermediateSize = "intermediate_size"
            case rmsNormEps = "rms_norm_eps"
            case hiddenAct = "hidden_act"
            case situBeta = "activation_situ_beta"
            case situLinearBeta = "activation_situ_linear_beta"
            case attentionResidualBlockSize = "attn_res_block_size"
            case qLoraRank = "q_lora_rank"
            case kvLoraRank = "kv_lora_rank"
            case qkNopeHeadDim = "qk_nope_head_dim"
            case qkRopeHeadDim = "qk_rope_head_dim"
            case vHeadDim = "v_head_dim"
            case mlaUseNope = "mla_use_nope"
            case mlaUseOutputGate = "mla_use_output_gate"
            case experts = "num_experts"
            case expertsPerToken = "num_experts_per_token"
            case sharedExperts = "num_shared_experts"
            case moeIntermediateSize = "moe_intermediate_size"
            case moeRouterActivation = "moe_router_activation_func"
            case moeRenormalize = "moe_renormalize"
            case routedScalingFactor = "routed_scaling_factor"
            case firstKDenseReplace = "first_k_dense_replace"
            case moeLayerFrequency = "moe_layer_freq"
            case expertGroups = "num_expert_group"
            case topkGroups = "topk_group"
            case routedExpertHiddenSize = "routed_expert_hidden_size"
            case latentMoEUseNorm = "latent_moe_use_norm"
            case tieWordEmbeddings = "tie_word_embeddings"
            case linearAttentionConfig = "linear_attn_config"
        }

        private struct LinearAttentionConfiguration: Decodable {
            var heads: Int?
            var headDim: Int?
            var convKernel: Int?
            var gateLowerBound: Float?
            var usesFullRankGate: Bool?
            var kdaLayers: [Int]?

            enum CodingKeys: String, CodingKey {
                case heads = "num_heads"
                case headDim = "head_dim"
                case convKernel = "short_conv_kernel_size"
                case gateLowerBound = "gate_lower_bound"
                case usesFullRankGate = "use_full_rank_gate"
                case kdaLayers = "kda_layers"
            }
        }

        init() {}

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)

            if let activation = try c.decodeIfPresent(String.self, forKey: .hiddenAct),
                activation != "situ"
            {
                throw DecodingError.dataCorruptedError(
                    forKey: .hiddenAct, in: c,
                    debugDescription: "Kimi K3 implements the situ activation only")
            }
            if let router = try c.decodeIfPresent(String.self, forKey: .moeRouterActivation),
                router != "sigmoid"
            {
                throw DecodingError.dataCorruptedError(
                    forKey: .moeRouterActivation, in: c,
                    debugDescription: "Kimi K3 implements sigmoid routing only")
            }

            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
            vocabularySize =
                try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? vocabularySize
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
            hiddenLayers =
                try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? hiddenLayers
            attentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? attentionHeads
            kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? kvHeads
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? intermediateSize
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? rmsNormEps
            situBeta = try c.decodeIfPresent(Float.self, forKey: .situBeta)
            situLinearBeta = try c.decodeIfPresent(Float.self, forKey: .situLinearBeta)
            attentionResidualBlockSize = try c.decodeIfPresent(
                Int.self, forKey: .attentionResidualBlockSize)
            qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank)
            kvLoraRank = try c.decodeIfPresent(Int.self, forKey: .kvLoraRank)
            qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim)
            qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim)
            vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim)
            mlaUseNope = try c.decodeIfPresent(Bool.self, forKey: .mlaUseNope) ?? mlaUseNope
            mlaUseOutputGate =
                try c.decodeIfPresent(Bool.self, forKey: .mlaUseOutputGate)
                ?? mlaUseOutputGate
            experts = try c.decodeIfPresent(Int.self, forKey: .experts)
            expertsPerToken =
                try c.decodeIfPresent(Int.self, forKey: .expertsPerToken) ?? expertsPerToken
            sharedExperts =
                try c.decodeIfPresent(Int.self, forKey: .sharedExperts) ?? sharedExperts
            moeIntermediateSize = try c.decodeIfPresent(
                Int.self, forKey: .moeIntermediateSize)
            moeRenormalize =
                try c.decodeIfPresent(Bool.self, forKey: .moeRenormalize) ?? moeRenormalize
            routedScalingFactor =
                try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor)
                ?? routedScalingFactor
            firstKDenseReplace =
                try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace)
                ?? firstKDenseReplace
            moeLayerFrequency =
                try c.decodeIfPresent(Int.self, forKey: .moeLayerFrequency)
                ?? moeLayerFrequency
            expertGroups =
                try c.decodeIfPresent(Int.self, forKey: .expertGroups) ?? expertGroups
            topkGroups = try c.decodeIfPresent(Int.self, forKey: .topkGroups) ?? topkGroups
            routedExpertHiddenSize = try c.decodeIfPresent(
                Int.self, forKey: .routedExpertHiddenSize)
            latentMoEUseNorm =
                try c.decodeIfPresent(Bool.self, forKey: .latentMoEUseNorm)
                ?? latentMoEUseNorm
            tieWordEmbeddings =
                try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
                ?? tieWordEmbeddings

            if let linear = try c.decodeIfPresent(
                LinearAttentionConfiguration.self, forKey: .linearAttentionConfig)
            {
                linearHeads = linear.heads ?? 0
                linearHeadDim = linear.headDim ?? 0
                linearConvKernel = linear.convKernel ?? linearConvKernel
                linearLowerBound = linear.gateLowerBound
                linearUsesFullRankGate = linear.usesFullRankGate ?? false
                kdaLayers = linear.kdaLayers ?? []
            }
        }

        public var beta: Float { situBeta ?? 1 }
        public var usesAttentionResidual: Bool { attentionResidualBlockSize != nil }

        /// `kda_layers` is one-based.
        public func isLinear(layerIndex: Int) -> Bool {
            kdaLayers.contains(layerIndex + 1)
        }

        public func isSparseMoE(layerIndex: Int) -> Bool {
            guard let experts, experts > 0, layerIndex >= firstKDenseReplace else {
                return false
            }
            return layerIndex % Swift.max(moeLayerFrequency, 1) == 0
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public var modelType: String = "moonvit3d"
        public var patchSize: Int = 14
        public var positionHeight: Int = 64
        public var positionWidth: Int = 64
        public var positionFrames: Int = 4
        public var heads: Int = 12
        public var hiddenLayers: Int = 27
        public var hiddenSize: Int = 1024
        public var intermediateSize: Int = 4096
        public var qkvHiddenSize: Int = 1536
        public var attentionBias: Bool = false
        public var linearBias: Bool = false
        public var patchEmbedBias: Bool = false
        public var mergeKernelSize: [Int] = [2, 2]
        public var multiModalHiddenSize: Int = 1024
        public var projectorEps: Float = 1e-5
        public var textHiddenSize: Int = 7168

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case patchSize = "patch_size"
            case positionHeight = "init_pos_emb_height"
            case positionWidth = "init_pos_emb_width"
            case positionFrames = "init_pos_emb_time"
            case heads = "vt_num_attention_heads"
            case hiddenLayers = "vt_num_hidden_layers"
            case hiddenSize = "vt_hidden_size"
            case intermediateSize = "vt_intermediate_size"
            case qkvHiddenSize = "qkv_hidden_size"
            case attentionBias = "attn_bias"
            case linearBias = "linear_bias"
            case patchEmbedBias = "patch_embed_proj_bias"
            case mergeKernelSize = "merge_kernel_size"
            case multiModalHiddenSize = "mm_hidden_size"
            case projectorEps = "projector_ln_eps"
            case textHiddenSize = "text_hidden_size"
        }

        init() {}

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? patchSize
            positionHeight =
                try c.decodeIfPresent(Int.self, forKey: .positionHeight) ?? positionHeight
            positionWidth =
                try c.decodeIfPresent(Int.self, forKey: .positionWidth) ?? positionWidth
            positionFrames =
                try c.decodeIfPresent(Int.self, forKey: .positionFrames) ?? positionFrames
            heads = try c.decodeIfPresent(Int.self, forKey: .heads) ?? heads
            hiddenLayers =
                try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? hiddenLayers
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? intermediateSize
            qkvHiddenSize =
                try c.decodeIfPresent(Int.self, forKey: .qkvHiddenSize) ?? qkvHiddenSize
            attentionBias =
                try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? attentionBias
            linearBias = try c.decodeIfPresent(Bool.self, forKey: .linearBias) ?? linearBias
            patchEmbedBias =
                try c.decodeIfPresent(Bool.self, forKey: .patchEmbedBias) ?? patchEmbedBias
            mergeKernelSize =
                try c.decodeIfPresent([Int].self, forKey: .mergeKernelSize)
                ?? mergeKernelSize
            multiModalHiddenSize =
                try c.decodeIfPresent(Int.self, forKey: .multiModalHiddenSize)
                ?? multiModalHiddenSize
            projectorEps =
                try c.decodeIfPresent(Float.self, forKey: .projectorEps) ?? projectorEps
            textHiddenSize =
                try c.decodeIfPresent(Int.self, forKey: .textHiddenSize) ?? textHiddenSize
        }

        public var headDim: Int {
            (qkvHiddenSize > 0 ? qkvHiddenSize : hiddenSize) / heads
        }
        public var mergeUnit: Int { mergeKernelSize[0] * mergeKernelSize[1] }
    }

    public var modelType: String
    public var textConfiguration: TextConfiguration
    public var visionConfiguration: VisionConfiguration
    public var imageTokenIndex: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case imageTokenIndex = "image_token_index"
        case mediaPlaceholderTokenID = "media_placeholder_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "kimi_k3"
        textConfiguration = try c.decode(TextConfiguration.self, forKey: .textConfiguration)
        visionConfiguration = try c.decode(
            VisionConfiguration.self, forKey: .visionConfiguration)
        let placeholder =
            try c.decodeIfPresent(Int.self, forKey: .mediaPlaceholderTokenID) ?? 163_605
        imageTokenIndex =
            try c.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? placeholder
    }
}

// MARK: - Text primitives

enum KimiK3Text {

    /// The `situ` activation.
    ///
    /// The gate is squashed by `beta * tanh(gate / beta) * sigmoid(gate)`, and the value
    /// gets its own optional `linear_beta * tanh(value / linear_beta)` squash before the
    /// product. Both are bounded, which is the point: a plain SwiGLU is not.
    static func situ(
        value: MLXArray, gate: MLXArray, beta: Float, linearBeta: Float?
    ) -> MLXArray {
        let dtype = value.dtype
        let gate32 = gate.asType(.float32)
        var value32 = value.asType(.float32)
        let scaled = beta * tanh(gate32 / beta) * sigmoid(gate32)
        if let linearBeta {
            value32 = linearBeta * tanh(value32 / linearBeta)
        }
        return (scaled * value32).asType(dtype)
    }

    /// Sigmoid-scored grouped routing.
    ///
    /// The correction bias steers the *selection* but never the returned weights, and the
    /// group score is the sum of each group's two best experts.
    static func groupExpertSelect(
        gates: MLXArray, bias: MLXArray?, topK: Int, groups: Int, topkGroups: Int,
        routedScalingFactor: Float, renormalize: Bool
    ) -> (indices: MLXArray, scores: MLXArray) {
        let inputType = gates.dtype
        let originalScores = sigmoid(gates.asType(.float32))
        var scores = originalScores
        if let bias {
            scores = scores + bias.asType(scores.dtype)
        }

        if groups > 1 {
            scores = unflatten(scores, axis: -1, shape: [groups, -1])
            let groupScores = top(scores, k: 2, axis: -1).sum(axis: -1, keepDims: true)
            let dropped = groups - topkGroups
            let groupIndices = argPartition(groupScores, kth: dropped - 1, axis: -2)[
                .ellipsis, ..<dropped, 0...]
            scores = putAlong(
                scores, groupIndices, values: MLXArray(Float(0)).asType(scores.dtype),
                axis: -2)
            scores = flattened(scores, start: -2, end: -1)
        }

        let indices = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var selected = takeAlong(originalScores, indices, axis: -1)
        if topK > 1, renormalize {
            selected = selected / (selected.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (indices, (selected * routedScalingFactor).asType(inputType))
    }

    private static let shortConvDecode = MLXFast.metalKernel(
        name: "kimi_short_conv_decode",
        inputNames: ["x", "history", "weight"],
        outputNames: ["out", "next"],
        source: """
            uint channel = thread_position_in_grid.x;
            uint batch = thread_position_in_grid.y;
            uint offset = batch * (K - 1) * C + channel;
            float value = 0.0f;
            for (uint tap = 0; tap < K - 1; ++tap) {
                value += static_cast<float>(weight[channel * K + tap])
                    * static_cast<float>(history[offset + tap * C]);
            }
            T input = x[batch * C + channel];
            value += static_cast<float>(weight[channel * K + K - 1]) * static_cast<float>(input);
            out[batch * C + channel] = static_cast<T>(value / (1.0f + metal::exp(-value)));
            for (uint tap = 0; tap < K - 2; ++tap) {
                next[offset + tap * C] = history[offset + (tap + 1) * C];
            }
            next[offset + (K - 2) * C] = input;
            """
    )

    /// A depthwise convolution over the last few positions, with its tail carried in the
    /// cache.
    final class ShortConv1d: Module {
        let kernelSize: Int
        let fuseDecode: Bool

        @ModuleInfo var conv: Conv1d

        init(channels: Int, kernelSize: Int, fuseDecode: Bool = false) {
            self.kernelSize = kernelSize
            self.fuseDecode = fuseDecode
            _conv.wrappedValue = Conv1d(
                inputChannels: channels, outputChannels: channels,
                kernelSize: kernelSize, padding: 0, groups: channels, bias: false)
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray, state: MLXArray?, mask: MLXArray?, lengths: MLXArray?
        ) -> (MLXArray, MLXArray) {
            if fuseDecode, !training, kernelSize > 1, x.dim(1) == 1,
                let state, mask == nil, lengths == nil, x.dtype == state.dtype,
                x.dtype == conv.weight.dtype, Device.defaultDevice().deviceType == .gpu
            {
                // The reference performs SiLU before rounding the convolution output.
                let channels = x.dim(-1)
                let result = KimiK3Text.shortConvDecode(
                    [x, state, conv.weight],
                    template: [("T", x.dtype), ("C", channels), ("K", kernelSize)],
                    grid: (channels, x.dim(0), 1), threadGroup: (min(channels, 1024), 1, 1),
                    outputShapes: [x.shape, state.shape], outputDTypes: [x.dtype, state.dtype])
                return (result[0], result[1])
            }
            var input = x
            if let mask {
                input = MLX.where(mask[.ellipsis, .newAxis], input, MLXArray(0))
            }

            let tail =
                state
                ?? MLXArray.zeros(
                    [input.dim(0), kernelSize - 1, input.dim(-1)], dtype: input.dtype)
            let padded = concatenated([tail, input], axis: 1)
            let out = silu(conv(padded))

            let keep = kernelSize - 1
            let newState: MLXArray
            if let lengths {
                let ends = clip(lengths, min: 0, max: input.dim(1))
                let positions = (ends[0..., .newAxis] + MLXArray(0 ..< keep))[
                    .ellipsis, .newAxis]
                newState = takeAlong(
                    padded,
                    broadcast(positions, to: [input.dim(0), keep, input.dim(-1)])
                        .asType(.int32), axis: 1)
            } else {
                newState = padded[0..., input.dim(1)...]
            }
            return (out, newState)
        }
    }

    /// The raw residuals saved at each block start, with the inverse RMS of each.
    ///
    /// One instance lives for the length of a forward and is threaded through the layers,
    /// which is how a block's mixture can read residuals from before it.
    final class ResidualBlocks {
        let eps: Float
        private(set) var raw: MLXArray?
        private(set) var inverseRMS: MLXArray?

        init(eps: Float) {
            self.eps = eps
        }

        func append(_ x: MLXArray) {
            let float = x.asType(.float32)
            let norm = rsqrt((float * float).mean(axis: -1) + eps)
                .expandedDimensions(axis: 0)
            let row = x.expandedDimensions(axis: 0)
            if raw == nil {
                raw = row
                inverseRMS = norm
            } else {
                raw = concatenated([raw!, row])
                inverseRMS = concatenated([inverseRMS!, norm])
            }
        }
    }

    /// A softmax mixture over the saved residuals and the running partial sum.
    ///
    /// Each candidate's logit is its projection onto one learned direction, scaled by its
    /// own inverse RMS -- so the mixture weighs directions, not magnitudes.
    static func mixResiduals(
        blocks: ResidualBlocks?, partialSum: MLXArray, weight: MLXArray, eps: Float
    ) -> MLXArray {
        guard let blocks, let raw = blocks.raw, let inverseRMS = blocks.inverseRMS else {
            return partialSum
        }
        let partial = partialSum.asType(.float32)
        let partialLogit =
            partial.matmul(weight)
            * rsqrt((partial * partial).mean(axis: -1) + eps)
        let logits = concatenated([
            raw.asType(.float32).matmul(weight) * inverseRMS,
            partialLogit.expandedDimensions(axis: 0),
        ])
        let probabilities = softmax(logits, axis: 0, precise: true)
        let mixed =
            (probabilities[..<(probabilities.dim(0) - 1), .ellipsis, .newAxis] * raw)
            .sum(axis: 0)
            + probabilities[probabilities.dim(0) - 1, .ellipsis, .newAxis] * partialSum
        return mixed.asType(partialSum.dtype)
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        let beta: Float
        let linearBeta: Float?

        init(_ config: KimiK3Configuration.TextConfiguration, intermediateSize: Int? = nil) {
            let hidden = intermediateSize ?? config.intermediateSize
            _gateProj.wrappedValue = Linear(config.hiddenSize, hidden, bias: false)
            _upProj.wrappedValue = Linear(config.hiddenSize, hidden, bias: false)
            _downProj.wrappedValue = Linear(hidden, config.hiddenSize, bias: false)
            beta = config.beta
            linearBeta = config.situLinearBeta
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(
                situ(value: upProj(x), gate: gateProj(x), beta: beta, linearBeta: linearBeta))
        }
    }

    /// The routed feed-forward, optionally through a narrower latent space.
    final class SparseMoE: Module, UnaryLayer {
        let config: KimiK3Configuration.TextConfiguration
        let latentSize: Int?

        @ModuleInfo var gate: Linear
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
        @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray
        @ModuleInfo(key: "routed_expert_down_proj") var routedDown: Linear?
        @ModuleInfo(key: "routed_expert_up_proj") var routedUp: Linear?
        @ModuleInfo(key: "routed_expert_norm") var routedNorm: RMSNorm?
        @ModuleInfo(key: "shared_experts") var sharedExperts: MLP?

        init(_ config: KimiK3Configuration.TextConfiguration) {
            self.config = config
            latentSize = config.routedExpertHiddenSize
            let experts = config.experts ?? 0
            let expertDim = latentSize ?? config.hiddenSize
            let beta = config.beta
            let linearBeta = config.situLinearBeta

            _gate.wrappedValue = Linear(config.hiddenSize, experts, bias: false)
            // `combine` is handed (gate, up); mlx-vlm's activation takes (up, gate).
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: expertDim,
                hiddenDims: config.moeIntermediateSize ?? config.intermediateSize,
                numExperts: experts,
                activationProduct: { gate, up in
                    situ(value: up, gate: gate, beta: beta, linearBeta: linearBeta)
                })
            _correctionBias.wrappedValue = MLXArray.zeros([experts], dtype: .float32)

            if let latentSize {
                _routedDown.wrappedValue = Linear(config.hiddenSize, latentSize, bias: false)
                _routedUp.wrappedValue = Linear(latentSize, config.hiddenSize, bias: false)
                if config.latentMoEUseNorm {
                    _routedNorm.wrappedValue = RMSNorm(
                        dimensions: latentSize, eps: config.rmsNormEps)
                }
            }
            if config.sharedExperts > 0 {
                _sharedExperts.wrappedValue = MLP(
                    config,
                    intermediateSize: (config.moeIntermediateSize ?? config.intermediateSize)
                        * config.sharedExperts)
            }
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (indices, weights) = groupExpertSelect(
                gates: gate(x), bias: correctionBias, topK: config.expertsPerToken,
                groups: config.expertGroups, topkGroups: config.topkGroups,
                routedScalingFactor: config.routedScalingFactor,
                renormalize: config.moeRenormalize)

            var y = routedDown.map { $0(x) } ?? x
            y = switchMLP(y, indices)
            y = (y * weights[.ellipsis, .newAxis]).sum(axis: -2)
            if let routedNorm {
                y = routedNorm(y)
            }
            if let routedUp {
                y = routedUp(y)
            }
            if let sharedExperts {
                y = y + sharedExperts(x)
            }
            return y
        }
    }
}

extension KimiK3Text {

    /// Kimi Delta Attention: a fused q/k/v short convolution feeding a delta-rule
    /// recurrence whose decay is per key dimension.
    ///
    /// The queries and keys are RMS-normalized with no weight and then scaled by powers of
    /// the head dimension -- the query twice, the key once -- rather than the attention
    /// applying a single scale later.
    final class DeltaAttention: Module {
        let heads: Int
        let headDim: Int
        let projectionDim: Int
        let convKernel: Int
        let scale: Float
        let lowerBound: Float?
        let usesFullRankGate: Bool

        @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
        @ModuleInfo(key: "qkv_conv") var qkvConv: ShortConv1d
        @ModuleInfo(key: "f_a_proj") var fAProj: Linear
        @ModuleInfo(key: "f_b_proj") var fBProj: Linear
        @ModuleInfo(key: "b_proj") var bProj: Linear
        @ModuleInfo(key: "g_proj") var gProj: Linear?
        @ModuleInfo(key: "g_a_proj") var gAProj: Linear?
        @ModuleInfo(key: "g_b_proj") var gBProj: Linear?
        @ModuleInfo(key: "o_norm") var oNorm: RMSNorm
        @ModuleInfo(key: "o_proj") var oProj: Linear

        @ParameterInfo(key: "A_log") var aLog: MLXArray
        @ParameterInfo(key: "dt_bias") var dtBias: MLXArray

        init(_ config: KimiK3Configuration.TextConfiguration) {
            heads = config.linearHeads
            headDim = config.linearHeadDim
            projectionDim = config.linearHeads * config.linearHeadDim
            convKernel = config.linearConvKernel
            scale = pow(Float(config.linearHeadDim), -0.5)
            lowerBound = config.linearLowerBound
            usesFullRankGate = config.linearUsesFullRankGate

            let hidden = config.hiddenSize
            _qkvProj.wrappedValue = Linear(hidden, 3 * projectionDim, bias: false)
            _qkvConv.wrappedValue = ShortConv1d(
                channels: 3 * projectionDim, kernelSize: convKernel)
            _fAProj.wrappedValue = Linear(hidden, headDim, bias: false)
            _fBProj.wrappedValue = Linear(headDim, projectionDim, bias: false)
            _bProj.wrappedValue = Linear(hidden, heads, bias: false)
            if usesFullRankGate {
                _gProj.wrappedValue = Linear(hidden, projectionDim, bias: false)
            } else {
                _gAProj.wrappedValue = Linear(hidden, headDim, bias: false)
                _gBProj.wrappedValue = Linear(headDim, projectionDim, bias: false)
            }
            _aLog.wrappedValue = MLXArray.zeros([heads])
            _dtBias.wrappedValue = MLXArray.zeros([projectionDim])
            _oNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
            _oProj.wrappedValue = Linear(projectionDim, hidden, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: MambaCache?) -> MLXArray {
            let (B, T) = (x.dim(0), x.dim(1))
            let P = projectionDim

            let (convolved, convState) = qkvConv(
                qkvProj(x), state: cache?[0], mask: mask, lengths: cache?.currentLengths)
            cache?[0] = convState

            var q = convolved[.ellipsis, ..<P].reshaped(B, T, heads, headDim)
            var k = convolved[.ellipsis, P ..< (2 * P)].reshaped(B, T, heads, headDim)
            let v = convolved[.ellipsis, (2 * P)...].reshaped(B, T, heads, headDim)

            let eps = 1e-6 / Float(headDim)
            q = (scale * scale) * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: eps)
            k = scale * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: eps)

            let a = fBProj(fAProj(x)).reshaped(B, T, heads, headDim)
            let b = bProj(x).reshaped(B, T, heads)

            let (out, state) = gatedDeltaUpdate(
                q: q, k: k, v: v, a: a, b: b,
                aLog: aLog.reshaped(heads, 1),
                dtBias: dtBias.reshaped(heads, headDim),
                state: cache?[1], mask: mask, lowerBound: lowerBound)
            if let cache {
                cache[1] = state
                cache.advance(T)
            }

            let gate =
                usesFullRankGate
                ? gProj!(x) : gBProj!(gAProj!(x))
            let gated =
                oNorm(out.reshaped(B, T, heads, headDim))
                * sigmoid(gate.reshaped(B, T, heads, headDim))
            return oProj(gated.reshaped(B, T, -1))
        }
    }

    /// Multi-head latent attention, NoPE for the content half with a separate positional
    /// key whose scores become the mask the content attention runs under.
    final class MLAAttention: Module {
        let heads: Int
        let qkNopeHeadDim: Int
        let qkRopeHeadDim: Int
        let qHeadDim: Int
        let vHeadDim: Int
        let kvLoraRank: Int
        let scale: Float
        let usesGate: Bool

        @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
        @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
        @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
        @ModuleInfo(key: "q_proj") var qProj: Linear?
        @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMQA: Linear
        @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
        @ModuleInfo(key: "embed_q") var embedQ: MultiLinear
        @ModuleInfo(key: "unembed_out") var unembedOut: MultiLinear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "g_proj") var gProj: Linear?

        init(_ config: KimiK3Configuration.TextConfiguration) {
            precondition(
                config.mlaUseNope, "Kimi K3 implements NoPE latent attention only")
            heads = config.attentionHeads
            qkNopeHeadDim = config.qkNopeHeadDim ?? 0
            qkRopeHeadDim = config.qkRopeHeadDim ?? 0
            qHeadDim = qkNopeHeadDim + qkRopeHeadDim
            vHeadDim = config.vHeadDim ?? 0
            kvLoraRank = config.kvLoraRank ?? 0
            scale = pow(Float(qHeadDim), -0.5)
            usesGate = config.mlaUseOutputGate

            let hidden = config.hiddenSize
            if let rank = config.qLoraRank {
                _qAProj.wrappedValue = Linear(hidden, rank, bias: false)
                _qALayerNorm.wrappedValue = RMSNorm(dimensions: rank, eps: 1e-6)
                _qBProj.wrappedValue = Linear(rank, heads * qHeadDim, bias: false)
            } else {
                _qProj.wrappedValue = Linear(hidden, heads * qHeadDim, bias: false)
            }
            _kvAProjWithMQA.wrappedValue = Linear(
                hidden, kvLoraRank + qkRopeHeadDim, bias: false)
            _kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: 1e-6)
            _embedQ.wrappedValue = MultiLinear(
                inputDimensions: qkNopeHeadDim, outputDimensions: kvLoraRank,
                numHeads: heads)
            _unembedOut.wrappedValue = MultiLinear(
                inputDimensions: kvLoraRank, outputDimensions: vHeadDim, numHeads: heads)
            _oProj.wrappedValue = Linear(heads * vHeadDim, hidden, bias: false)
            if usesGate {
                _gProj.wrappedValue = Linear(hidden, heads * vHeadDim, bias: false)
            }
            super.init()
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var q =
                qBProj != nil
                ? qBProj!(qALayerNorm!(qAProj!(x)))
                : qProj!(x)
            q = q.reshaped(B, L, heads, qHeadDim).transposed(0, 2, 1, 3)
            let queryParts = split(q, indices: [qkNopeHeadDim], axis: -1)
            var queryNope = queryParts[0]
            let queryPE = queryParts[1]

            let compressed = kvAProjWithMQA(x)
            let kvParts = split(compressed, indices: [kvLoraRank], axis: -1)
            var latent = kvALayerNorm(kvParts[0]).expandedDimensions(axis: 1)
            var keyPE = kvParts[1]
                .reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

            if let cache {
                (latent, keyPE) = cache.update(keys: latent, values: keyPE)
            }

            // The positional half is scored on its own and handed down as the mask, which
            // is what lets the content half run against the compressed latent.
            var positionalScores = (queryPE * scale).matmul(keyPE.swappedAxes(-1, -2))
            if let mask {
                positionalScores = MLX.where(
                    mask, positionalScores,
                    MLXArray(-Float.greatestFiniteMagnitude, dtype: positionalScores.dtype))
            }

            let absorbed =
                L == 1
                || L
                    <= maxAbsorbedQueries(
                        kvLoraRank: kvLoraRank, qkNopeHeadDim: qkNopeHeadDim,
                        vHeadDim: vHeadDim, cacheLength: latentLength(latent))

            let keys: MLXArray
            let values: MLXArray
            if absorbed {
                queryNope = embedQ(queryNope)
                keys = latent
                values = latent
            } else {
                keys = embedQ(latent, transpose: false)
                values = unembedOut(latent)
            }

            var output = MLXFast.scaledDotProductAttention(
                queries: queryNope, keys: keys, values: values, scale: scale,
                mask: .array(positionalScores))
            if absorbed {
                output = unembedOut(output)
            }

            output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
            if usesGate, let gProj {
                output = output * sigmoid(gProj(x))
            }
            return oProj(output)
        }
    }

    /// The two shapes a layer's `self_attn` slot can take. One property, one checkpoint key.
    protocol AttentionLayer: Module {
        func attend(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray
    }

    final class DecoderLayer: Module {
        let isLinear: Bool
        let eps: Float
        let usesAttentionResidual: Bool
        let isBlockStart: Bool

        @ModuleInfo(key: "self_attn") var attention: AttentionLayer
        @ModuleInfo var mlp: UnaryLayer
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        @ModuleInfo(key: "self_attention_res_proj") var attentionResidualProj: Linear?
        @ModuleInfo(key: "self_attention_res_norm") var attentionResidualNorm: RMSNorm?
        @ModuleInfo(key: "mlp_res_proj") var mlpResidualProj: Linear?
        @ModuleInfo(key: "mlp_res_norm") var mlpResidualNorm: RMSNorm?

        init(_ config: KimiK3Configuration.TextConfiguration, layerIndex: Int) {
            isLinear = config.isLinear(layerIndex: layerIndex)
            eps = config.rmsNormEps
            usesAttentionResidual = config.usesAttentionResidual
            isBlockStart =
                config.attentionResidualBlockSize.map {
                    layerIndex % Swift.max($0, 1) == 0
                } ?? false

            _attention.wrappedValue =
                isLinear ? DeltaAttention(config) : MLAAttention(config)
            _mlp.wrappedValue =
                config.isSparseMoE(layerIndex: layerIndex)
                ? SparseMoE(config) : MLP(config)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)

            if usesAttentionResidual {
                _attentionResidualProj.wrappedValue = Linear(
                    config.hiddenSize, 1, bias: false)
                _attentionResidualNorm.wrappedValue = RMSNorm(
                    dimensions: config.hiddenSize, eps: config.rmsNormEps)
                _mlpResidualProj.wrappedValue = Linear(config.hiddenSize, 1, bias: false)
                _mlpResidualNorm.wrappedValue = RMSNorm(
                    dimensions: config.hiddenSize, eps: config.rmsNormEps)
            }
            super.init()
        }

        /// The mixture's direction is the residual norm's weight times the projection's,
        /// so the two collapse into one vector.
        private func effectiveWeight(norm: RMSNorm?, projection: Linear?) -> MLXArray {
            norm!.weight.asType(.float32) * projection!.weight.reshaped(-1)
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXArray?, cache: KVCache?, blocks: ResidualBlocks?
        ) -> MLXArray {
            guard usesAttentionResidual, let blocks else {
                let h = x + attention.attend(inputLayerNorm(x), mask: mask, cache: cache)
                return h + mlp(postAttentionLayerNorm(h))
            }

            var partialSum: MLXArray? = x
            var h = mixResiduals(
                blocks: blocks, partialSum: partialSum!,
                weight: effectiveWeight(
                    norm: attentionResidualNorm, projection: attentionResidualProj),
                eps: eps)
            if isBlockStart {
                blocks.append(partialSum!)
                partialSum = nil
            }

            let attended = attention.attend(inputLayerNorm(h), mask: mask, cache: cache)
            partialSum = partialSum.map { $0 + attended } ?? attended

            h = mixResiduals(
                blocks: blocks, partialSum: partialSum!,
                weight: effectiveWeight(
                    norm: mlpResidualNorm, projection: mlpResidualProj),
                eps: eps)
            return partialSum! + mlp(postAttentionLayerNorm(h))
        }
    }
}

extension KimiK3Text.DeltaAttention: KimiK3Text.AttentionLayer {
    func attend(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        callAsFunction(x, mask: mask, cache: cache as? MambaCache)
    }
}

extension KimiK3Text.MLAAttention: KimiK3Text.AttentionLayer {
    func attend(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        callAsFunction(x, mask: mask, cache: cache)
    }
}

extension KimiK3Text {

    final class Backbone: Module {
        let config: KimiK3Configuration.TextConfiguration
        let recurrentIndex: Int?
        let attentionIndex: Int?

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo var layers: [DecoderLayer]
        @ModuleInfo var norm: RMSNorm
        @ModuleInfo(key: "output_attn_res_proj") var outputResidualProj: Linear?
        @ModuleInfo(key: "output_attn_res_norm") var outputResidualNorm: RMSNorm?

        init(_ config: KimiK3Configuration.TextConfiguration) {
            self.config = config
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
                DecoderLayer(config, layerIndex: $0)
            }
            _norm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            if config.usesAttentionResidual {
                _outputResidualProj.wrappedValue = Linear(config.hiddenSize, 1, bias: false)
                _outputResidualNorm.wrappedValue = RMSNorm(
                    dimensions: config.hiddenSize, eps: config.rmsNormEps)
            }
            recurrentIndex = config.kdaLayers.first.map { $0 - 1 }
            attentionIndex = (0 ..< config.hiddenLayers).first {
                !config.isLinear(layerIndex: $0)
            }
            super.init()
        }

        func callAsFunction(
            inputs: MLXArray?, inputEmbedding: MLXArray?, cache: [KVCache]?
        ) -> MLXArray {
            guard var h = inputEmbedding ?? inputs.map({ embedTokens($0) }) else {
                fatalError("either inputs or inputEmbedding must be provided")
            }

            let recurrentMask = recurrentIndex.flatMap { index -> MLXArray? in
                guard let cache, cache.indices.contains(index) else { return nil }
                return createSSMMask(h: h, cache: cache[index] as? MambaCache)
            }
            var attentionMask: MLXArray?
            if let attentionIndex, h.dim(1) > 1 {
                let entry = cache.flatMap {
                    $0.indices.contains(attentionIndex) ? $0[attentionIndex] : nil
                }
                let mode = createAttentionMask(h: h, cache: entry, returnArray: true)
                if case .array(let value) = mode { attentionMask = value }
            }

            let blocks =
                config.usesAttentionResidual ? ResidualBlocks(eps: config.rmsNormEps) : nil
            for (index, layer) in layers.enumerated() {
                h = layer(
                    h, mask: layer.isLinear ? recurrentMask : attentionMask,
                    cache: cache?[index], blocks: blocks)
            }

            if let blocks, let outputResidualNorm, let outputResidualProj {
                h = mixResiduals(
                    blocks: blocks, partialSum: h,
                    weight: outputResidualNorm.weight.asType(.float32)
                        * outputResidualProj.weight.reshaped(-1),
                    eps: config.rmsNormEps)
            }
            return norm(h)
        }
    }
}

// MARK: - Vision

enum KimiK3Vision {

    /// Bilinear resampling with half-pixel centers, which is what a torch interpolate with
    /// `align_corners=False` does.
    static func bilinearResample(_ weight: MLXArray, height: Int, width: Int) -> MLXArray {
        let (H, W) = (weight.dim(0), weight.dim(1))
        if H == height, W == width { return weight }

        func indices(_ inputSize: Int, _ outputSize: Int) -> (MLXArray, MLXArray, MLXArray) {
            var source =
                (MLXArray(0 ..< outputSize).asType(.float32) + 0.5)
                * (Float(inputSize) / Float(outputSize)) - 0.5
            source = clip(source, min: 0, max: Float(inputSize - 1))
            let low = floor(source).asType(.int32)
            let high = minimum(low + 1, inputSize - 1)
            return (low, high, source - low.asType(.float32))
        }

        let table = weight.asType(.float32)
        let (r0, r1, rf) = indices(H, height)
        let rows =
            table[r0] * (1 - rf)[0..., .newAxis, .newAxis]
            + table[r1] * rf[0..., .newAxis, .newAxis]
        let (c0, c1, cf) = indices(W, width)
        return rows[0..., c0] * (1 - cf)[.newAxis, 0..., .newAxis]
            + rows[0..., c1] * cf[.newAxis, 0..., .newAxis]
    }

    /// The fixed sinusoidal table the frame axis adds on top of the learned 2-D one.
    static func timeEmbedding(dimensions: Int, frames: Int) -> MLXArray {
        let omega =
            1
            / pow(
                MLXArray(Float(10000)),
                MLXArray(0 ..< (dimensions / 2)).asType(.float32)
                    / Float(dimensions / 2))
        let positions = MLXArray(0 ..< frames).asType(.float32)
        let out = positions[0..., .newAxis] * omega[.newAxis, 0...]
        return concatenated([sin(out), cos(out)], axis: 1)
    }

    /// Interleaved 2-D rotary angles: the x and y frequencies alternate rather than
    /// occupying separate halves.
    static func rope2DAngles(headDim: Int, frames: Int, height: Int, width: Int)
        -> MLXArray
    {
        let quarters = headDim / 4
        let frequencies =
            1
            / pow(
                MLXArray(Float(10000)),
                MLXArray(stride(from: 0, to: headDim, by: 4).map(Float.init))
                    .asType(.float32)[..<quarters] / Float(headDim))
        let y = repeated(MLXArray(0 ..< height).asType(.float32), count: width)
        let x = tiled(MLXArray(0 ..< width).asType(.float32), repetitions: [height])
        let xFrequencies = x[0..., .newAxis] * frequencies[.newAxis, 0...]
        let yFrequencies = y[0..., .newAxis] * frequencies[.newAxis, 0...]
        var angles = MLX.stacked([xFrequencies, yFrequencies], axis: -1)
            .reshaped(height * width, -1)
        if frames > 1 {
            angles = tiled(angles, repetitions: [frames, 1])
        }
        return angles
    }

    static func applyRoPE2D(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let value = x.asType(.float32)
        let even = value[.ellipsis, .stride(by: 2)]
        let odd = value[.ellipsis, .stride(from: 1, by: 2)]
        let c = cos[0..., .newAxis, 0...]
        let s = sin[0..., .newAxis, 0...]
        return MLX.stacked([even * c - odd * s, even * s + odd * c], axis: -1)
            .flattened(start: -2)
            .asType(dtype)
    }

    /// The learned 2-D grid, resampled per image and offset per frame.
    final class PositionEmbedding: Module {
        let height: Int
        let width: Int
        let frames: Int

        @ParameterInfo(key: "weight") var weight: MLXArray
        let timeWeight: [Float]
        let dimensions: Int

        init(_ config: KimiK3Configuration.VisionConfiguration) {
            height = config.positionHeight
            width = config.positionWidth
            frames = config.positionFrames
            dimensions = config.hiddenSize
            _weight.wrappedValue = MLXArray.zeros([height, width, config.hiddenSize])
            let table = timeEmbedding(dimensions: config.hiddenSize, frames: frames)
            eval(table)
            timeWeight = table.asType(.float32).asArray(Float.self)
            super.init()
        }

        func callAsFunction(_ x: MLXArray, grid: [[Int]]) -> MLXArray {
            var rows: [MLXArray] = []
            for entry in grid {
                let (t, h, w) = (entry[0], entry[1], entry[2])
                let plane = bilinearResample(weight, height: h, width: w)
                    .reshaped(h * w, -1)
                if t == 1 {
                    rows.append(plane)
                } else {
                    let time = MLXArray(
                        Array(timeWeight[0 ..< (t * dimensions)]), [t, dimensions])
                    rows.append(
                        (plane[.newAxis] + time[0..., .newAxis, 0...])
                            .reshaped(t * h * w, -1))
                }
            }
            return x + concatenated(rows).asType(x.dtype)
        }
    }

    final class PatchEmbed: Module {
        @ModuleInfo var proj: Conv2d
        @ModuleInfo(key: "pos_emb") var positionEmbedding: PositionEmbedding

        init(_ config: KimiK3Configuration.VisionConfiguration) {
            _proj.wrappedValue = Conv2d(
                inputChannels: 3, outputChannels: config.hiddenSize,
                kernelSize: .init(config.patchSize), stride: .init(config.patchSize),
                bias: config.patchEmbedBias)
            _positionEmbedding.wrappedValue = PositionEmbedding(config)
            super.init()
        }

        func callAsFunction(_ patches: MLXArray, grid: [[Int]]) -> MLXArray {
            let embedded = proj(patches).reshaped(patches.dim(0), -1)
            return positionEmbedding(embedded, grid: grid)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo var fc0: Linear
        @ModuleInfo var fc1: Linear

        init(_ config: KimiK3Configuration.VisionConfiguration) {
            _fc0.wrappedValue = Linear(
                config.hiddenSize, config.intermediateSize, bias: config.linearBias)
            _fc1.wrappedValue = Linear(
                config.intermediateSize, config.hiddenSize, bias: config.linearBias)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc1(geluApproximate(fc0(x)))
        }
    }

    /// A pre-norm block over a flat patch sequence. Both norms use an epsilon of 2^-7,
    /// which is far larger than the usual 1e-5 and part of the architecture.
    final class Block: Module {
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo var norm0: RMSNorm
        @ModuleInfo var norm1: RMSNorm
        @ModuleInfo var wqkv: Linear
        @ModuleInfo var wo: Linear
        @ModuleInfo var mlp: MLP

        init(_ config: KimiK3Configuration.VisionConfiguration) {
            heads = config.heads
            let qkvHidden = config.qkvHiddenSize > 0 ? config.qkvHiddenSize : config.hiddenSize
            headDim = qkvHidden / config.heads
            scale = pow(Float(headDim), -0.5)

            _norm0.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: pow(2, -7))
            _norm1.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: pow(2, -7))
            _wqkv.wrappedValue = Linear(
                config.hiddenSize, 3 * qkvHidden, bias: config.attentionBias)
            _wo.wrappedValue = Linear(
                qkvHidden, config.hiddenSize, bias: config.attentionBias)
            _mlp.wrappedValue = MLP(config)
            super.init()
        }

        func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
            let length = x.dim(0)
            let fused = wqkv(norm0(x)).reshaped(length, 3, heads, headDim)
            var q = fused[0..., 0]
            var k = fused[0..., 1]
            let v = fused[0..., 2]
            q = applyRoPE2D(q, cos: cos, sin: sin)
            k = applyRoPE2D(k, cos: cos, sin: sin)

            let out = MLXFast.scaledDotProductAttention(
                queries: q.transposed(1, 0, 2)[.newAxis, .ellipsis],
                keys: k.transposed(1, 0, 2)[.newAxis, .ellipsis],
                values: v.transposed(1, 0, 2)[.newAxis, .ellipsis],
                scale: scale, mask: .none)
            let attended = out[0].transposed(1, 0, 2).reshaped(length, -1)
            let h = x + wo(attended)
            return h + mlp(norm1(h))
        }
    }

    final class Encoder: Module {
        let headDim: Int

        @ModuleInfo var blocks: [Block]
        @ModuleInfo(key: "final_layernorm") var finalLayerNorm: RMSNorm

        init(_ config: KimiK3Configuration.VisionConfiguration) {
            headDim = config.headDim
            _blocks.wrappedValue = (0 ..< config.hiddenLayers).map { _ in Block(config) }
            _finalLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: pow(2, -7))
            super.init()
        }

        func callAsFunction(_ x: MLXArray, frames: Int, height: Int, width: Int)
            -> MLXArray
        {
            let angles = rope2DAngles(
                headDim: headDim, frames: frames, height: height, width: width)
            let cosine = cos(angles)
            let sine = sin(angles)
            var h = x
            for block in blocks {
                h = block(h, cos: cosine, sin: sine)
            }
            return finalLayerNorm(h)
        }
    }

    /// Averages the frame axis away and groups each merge block's patches into one row,
    /// leaving the projector to flatten them.
    static func poolAndMerge(
        _ x: MLXArray, frames: Int, height: Int, width: Int, kernel: [Int]
    ) -> MLXArray {
        let (kh, kw) = (kernel[0], kernel[1])
        let dimensions = x.dim(-1)
        let (nh, nw) = (height / kh, width / kw)
        return
            x
            .reshaped(frames, nh, kh, nw, kw, dimensions)
            .transposed(0, 1, 3, 2, 4, 5)
            .mean(axis: 0)
            .reshaped(nh * nw, kh * kw, dimensions)
    }

    final class VisionModel: Module {
        let config: KimiK3Configuration.VisionConfiguration

        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
        @ModuleInfo var encoder: Encoder

        init(_ config: KimiK3Configuration.VisionConfiguration) {
            precondition(
                config.modelType == "moonvit3d",
                "Unsupported vision modelType: \(config.modelType)")
            self.config = config
            _patchEmbed.wrappedValue = PatchEmbed(config)
            _encoder.wrappedValue = Encoder(config)
            super.init()
        }

        func callAsFunction(_ patches: MLXArray, grid: [[Int]]) -> MLXArray {
            let normalized = grid.map { $0.count == 2 ? [1, $0[0], $0[1]] : $0 }
            let embedded = patchEmbed(patches, grid: normalized)

            var outputs: [MLXArray] = []
            var start = 0
            for entry in normalized {
                let (t, h, w) = (entry[0], entry[1], entry[2])
                let count = t * h * w
                let encoded = encoder(
                    embedded[start ..< (start + count)], frames: t, height: h, width: w)
                outputs.append(
                    poolAndMerge(
                        encoded, frames: t, height: h, width: w,
                        kernel: config.mergeKernelSize))
                start += count
            }
            return concatenated(outputs, axis: 0)
        }
    }

    /// Flattens each merge block's rows and projects them into the text width.
    final class Projector: Module {
        let width: Int

        @ModuleInfo var proj: [Module]
        @ModuleInfo(key: "post_norm") var postNorm: RMSNorm

        init(_ config: KimiK3Configuration) {
            let vision = config.visionConfiguration
            width = vision.multiModalHiddenSize * vision.mergeUnit
            // The activation keeps index 1 so the linears stay at proj.0 and proj.2.
            _proj.wrappedValue = [
                Linear(width, width, bias: false),
                GELU(),
                Linear(width, config.textConfiguration.hiddenSize, bias: false),
            ]
            _postNorm.wrappedValue = RMSNorm(
                dimensions: config.textConfiguration.hiddenSize, eps: vision.projectorEps)
            super.init()
        }

        func callAsFunction(_ features: MLXArray) -> MLXArray {
            var h = features.reshaped(features.dim(0), -1)
            for layer in proj {
                if let linear = layer as? Linear {
                    h = linear(h)
                } else if let activation = layer as? GELU {
                    h = activation(h)
                }
            }
            return postNorm(h)
        }
    }
}

// MARK: - Model

public final class KimiK3: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") var visionTower: KimiK3Vision.VisionModel
    @ModuleInfo(key: "language_model") var languageModel: LanguageHalf
    @ModuleInfo(key: "mm_projector") var projector: KimiK3Vision.Projector

    final class LanguageHalf: Module {
        @ModuleInfo var model: KimiK3Text.Backbone
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        init(_ config: KimiK3Configuration.TextConfiguration) {
            _model.wrappedValue = KimiK3Text.Backbone(config)
            if !config.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.hiddenSize, config.vocabularySize, bias: false)
            }
            super.init()
        }
    }

    public let config: KimiK3Configuration

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] {
        Array(
            repeating: config.textConfiguration.kvHeads,
            count: config.textConfiguration.hiddenLayers)
    }

    public init(_ config: KimiK3Configuration) {
        self.config = config
        _visionTower.wrappedValue = KimiK3Vision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = LanguageHalf(config.textConfiguration)
        _projector.wrappedValue = KimiK3Vision.Projector(config)
    }

    /// Two slots for the recurrent layers, one attention cache for the rest.
    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        try languageModel.model.layers.map { layer in
            layer.isLinear
                ? MambaCache() : try makeAttentionKVCache(configuration: cacheConfiguration)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(inputs: inputs, inputEmbedding: nil, cache: cache)
    }

    private func logits(inputs: MLXArray?, inputEmbedding: MLXArray?, cache: [KVCache]?)
        -> MLXArray
    {
        let hidden = languageModel.model(
            inputs: inputs, inputEmbedding: inputEmbedding, cache: cache)
        if let head = languageModel.lmHead {
            return head(hidden)
        }
        return languageModel.model.embedTokens.asLinear(hidden)
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
            let hidden = visionTower(
                image.pixels.transposed(0, 2, 3, 1)
                    .asType(visionTower.patchEmbed.proj.weight.dtype), grid: grid)
            let features = projector(hidden)
            return Self.scatter(
                embedded: languageModel.model.embedTokens(tokens),
                features: features.asType(
                    languageModel.model.embedTokens.weight.dtype),
                selected: tokens .== MLXArray(Int32(config.imageTokenIndex)))
        }

        let total = embeddings?.dim(1) ?? tokens.dim(1)

        func forward(_ range: Range<Int>) -> MLXArray {
            if let embeddings {
                return logits(
                    inputs: nil, inputEmbedding: embeddings[0..., range], cache: cache)
            }
            return logits(inputs: tokens[0..., range], inputEmbedding: nil, cache: cache)
        }

        let processed = try prefill.forEachChunk(total: total) { range in
            _ = forward(range)
            asyncEval(cache)
        }
        if processed > 0 { eval(cache) }
        prefill.progress?(total, total)
        return .logits(LMOutput(logits: forward(processed ..< total)))
    }

    private static func scatter(
        embedded: MLXArray, features: MLXArray, selected: MLXArray
    ) -> MLXArray {
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

    /// Moves the checkpoint's names onto the module tree.
    ///
    /// The residual blocks store their projection and norm as bare parameters rather than
    /// submodules, the routed experts live under `block_sparse_moe` one per key, the
    /// linear attention stores its three projections and convolutions separately, and the
    /// latent attention stores one fused `kv_b_proj` the module wants as two.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let text = config.textConfiguration
        let prefix = "language_model."
        var other: [String: MLXArray] = [:]
        var inner: [String: MLXArray] = [:]

        for (key, value) in weights {
            if key.hasPrefix(prefix) {
                inner[String(key.dropFirst(prefix.count))] = value
            } else {
                other[key] = value
            }
        }

        inner = inner.filter { !$0.key.hasPrefix("model.mtp") }
        inner = inner.filter { key, _ in
            let parts = key.split(separator: ".")
            guard parts.count >= 3, parts[0] == "model", parts[1] == "layers",
                let layer = Int(parts[2])
            else { return true }
            return layer < text.hiddenLayers
        }
        if text.tieWordEmbeddings {
            inner.removeValue(forKey: "lm_head.weight")
        }

        for source in ["self_attention_res", "mlp_res", "output_attn_res"] {
            for (suffix, replacement) in [
                (".proj_weight", "_proj.weight"), (".norm_weight", "_norm.weight"),
            ] {
                let full = source + suffix
                for key in inner.keys where key.hasSuffix(full) {
                    inner[String(key.dropLast(full.count)) + source + replacement] =
                        inner.removeValue(forKey: key)
                }
            }
        }

        for (index, layer) in languageModel.model.layers.enumerated() {
            let path = "model.layers.\(index)"
            if layer.mlp is KimiK3Text.SparseMoE {
                Self.moveExperts(
                    &inner, source: "\(path).block_sparse_moe",
                    destination: "\(path).mlp", experts: text.experts ?? 0)
            }

            let attention = "\(path).self_attn"
            if layer.isLinear {
                Self.fuseLinearAttention(&inner, path: attention, config: text)
            }
            Self.splitFusedKV(&inner, path: attention, config: text)
        }

        for (key, value) in inner {
            other[prefix + key] = value
        }

        // The tower's patch convolution comes in the torch layout.
        for (key, value) in other
        where key.contains("patch_embed.proj.weight") && value.ndim == 4
            && !Self.isMLXConvolution(value)
        {
            other[key] = value.transposed(0, 2, 3, 1)
        }
        return other
    }

    private static func moveExperts(
        _ weights: inout [String: MLXArray], source: String, destination: String,
        experts: Int
    ) {
        for (from, to) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
            for suffix in ["weight", "scales", "biases"] {
                guard weights["\(source).experts.0.\(from).\(suffix)"] != nil else {
                    continue
                }
                var stacked: [MLXArray] = []
                for expert in 0 ..< experts {
                    guard
                        let value = weights.removeValue(
                            forKey: "\(source).experts.\(expert).\(from).\(suffix)")
                    else { break }
                    stacked.append(value)
                }
                if stacked.count == experts {
                    weights["\(destination).switch_mlp.\(to).\(suffix)"] = MLX.stacked(
                        stacked)
                }
            }
        }

        for name in [
            "shared_experts.gate_proj", "shared_experts.up_proj",
            "shared_experts.down_proj", "routed_expert_down_proj",
            "routed_expert_up_proj", "routed_expert_norm", "gate",
            "switch_mlp.gate_proj", "switch_mlp.up_proj", "switch_mlp.down_proj",
        ] {
            for suffix in ["weight", "scales", "biases"] {
                if let value = weights.removeValue(forKey: "\(source).\(name).\(suffix)") {
                    weights["\(destination).\(name).\(suffix)"] = value
                }
            }
        }

        for key in ["\(source).gate.e_score_correction_bias", "\(source).e_score_correction_bias"] {
            if let value = weights.removeValue(forKey: key) {
                weights["\(destination).e_score_correction_bias"] = value
            }
        }
    }

    private static func fuseLinearAttention(
        _ weights: inout [String: MLXArray], path: String,
        config: KimiK3Configuration.TextConfiguration
    ) {
        for (from, to) in [("q_conv1d", "q_conv"), ("k_conv1d", "k_conv"), ("v_conv1d", "v_conv")] {
            if var value = weights.removeValue(forKey: "\(path).\(from).weight") {
                if value.ndim == 3 {
                    value = value.movedAxis(source: 2, destination: 1)
                }
                weights["\(path).\(to).conv.weight"] = value
            }
        }

        for name in ["dt_bias", "A_log"] {
            if let value = weights["\(path).\(name)"], value.ndim > 1 {
                weights["\(path).\(name)"] = value.reshaped(-1)
            }
        }
        // A checkpoint can carry more decay rates than the configuration's head count.
        if let value = weights["\(path).A_log"], value.dim(0) > config.linearHeads {
            weights["\(path).A_log"] = value[..<config.linearHeads]
        }

        if weights["\(path).qkv_proj.weight"] == nil {
            for suffix in ["weight", "scales", "biases"] {
                let parts = ["q", "k", "v"].map { "\(path).\($0)_proj.\(suffix)" }
                if parts.allSatisfy({ weights[$0] != nil }) {
                    weights["\(path).qkv_proj.\(suffix)"] = concatenated(
                        parts.map { weights.removeValue(forKey: $0)! }, axis: 0)
                }
            }
        }

        let convolutions = ["q", "k", "v"].map { "\(path).\($0)_conv.conv.weight" }
        if weights["\(path).qkv_conv.conv.weight"] == nil,
            convolutions.allSatisfy({ weights[$0] != nil })
        {
            weights["\(path).qkv_conv.conv.weight"] = concatenated(
                convolutions.map { weights.removeValue(forKey: $0)! }, axis: 0)
        }
    }

    private static func splitFusedKV(
        _ weights: inout [String: MLXArray], path: String,
        config: KimiK3Configuration.TextConfiguration
    ) {
        guard var fused = weights.removeValue(forKey: "\(path).kv_b_proj.weight") else {
            return
        }
        let qkNope = config.qkNopeHeadDim ?? 0
        let headDim = qkNope + (config.vHeadDim ?? 0)

        var bits: Int?
        var groupSize: Int?
        if let scales = weights.removeValue(forKey: "\(path).kv_b_proj.scales") {
            let biases = weights.removeValue(forKey: "\(path).kv_b_proj.biases")
            let rank = config.kvLoraRank ?? 1
            bits = (fused.dim(-1) * 32) / rank
            groupSize = rank / scales.dim(-1)
            fused = dequantized(
                fused, scales: scales, biases: biases, groupSize: groupSize!, bits: bits!)
        }

        let reshaped = fused.reshaped(config.attentionHeads, headDim, -1)
        var embedQ = contiguous(reshaped[0..., ..<qkNope, 0...].swappedAxes(-1, -2))
        var unembedOut = contiguous(reshaped[0..., qkNope..., 0...])

        if let bits, let groupSize {
            let (qWeight, qScales, qBiases) = MLX.quantized(
                embedQ, groupSize: groupSize, bits: bits)
            let (vWeight, vScales, vBiases) = MLX.quantized(
                unembedOut, groupSize: groupSize, bits: bits)
            weights["\(path).embed_q.scales"] = qScales
            weights["\(path).embed_q.biases"] = qBiases
            weights["\(path).unembed_out.scales"] = vScales
            weights["\(path).unembed_out.biases"] = vBiases
            embedQ = qWeight
            unembedOut = vWeight
        }
        weights["\(path).embed_q.weight"] = embedQ
        weights["\(path).unembed_out.weight"] = unembedOut
    }

    private static func isMLXConvolution(_ array: MLXArray) -> Bool {
        guard array.ndim == 4 else { return false }
        let (out, kH, kW) = (array.dim(0), array.dim(1), array.dim(2))
        return out >= kH && out >= kW && kH == kW
    }
}

extension KimiK3: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
