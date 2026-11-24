//
//  NemotronH.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/nemotron_h.py

import Foundation

public struct NemotronHConfiguration: Codable, Sendable {
    // Selected by speech wrappers whose checkpoints require FP32 recurrent dynamics.
    var speechPrecision = false
    /// Puzzle checkpoints override routed expert dimensions independently per block.
    public struct BlockConfiguration: Codable, Sendable {
        public var blockType: String
        public var nRoutedExperts: Int?
        public var moeIntermediateSize: Int?
        public var numExpertsPerTok: Int?

        enum CodingKeys: String, CodingKey {
            case blockType = "block_type"
            case nRoutedExperts = "n_routed_experts"
            case moeIntermediateSize = "moe_intermediate_size"
            case numExpertsPerTok = "num_experts_per_tok"
        }
    }

    public var blockConfigurations: [BlockConfiguration]? = nil
    public var moeLatentSize: Int? = nil
    public var numNextNPredictLayers: Int = 0
    public var modelType: String = "nemotron_h"
    public var vocabSize: Int
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var attentionBias: Bool
    public var mambaNumHeads: Int
    public var mambaHeadDim: Int
    public var mambaProjBias: Bool
    public var ssmStateSize: Int
    public var convKernel: Int
    public var nGroups: Int
    public var intermediateSize: Int
    public var moeIntermediateSize: Int
    public var moeSharedExpertIntermediateSize: Int
    public var nRoutedExperts: Int
    public var nSharedExperts: Int?
    public var numExpertsPerTok: Int
    public var hybridOverridePattern: String
    public var layerNormEpsilon: Float
    public var mlpBias: Bool
    public var useBias: Bool
    public var useConvBias: Bool
    public var tieWordEmbeddings: Bool
    public var ropeTheta: Float
    public var headDim: Int?
    public var nGroup: Int
    public var topkGroup: Int
    public var normTopkProb: Bool
    public var routedScalingFactor: Float
    public var timeStepLimitMin: Float
    public var timeStepLimitMax: Float

    enum CodingKeys: String, CodingKey {
        case blockConfigurations = "block_configs"
        case moeLatentSize = "moe_latent_size"
        case numNextNPredictLayers = "num_nextn_predict_layers"
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case attentionBias = "attention_bias"
        case mambaNumHeads = "mamba_num_heads"
        case mambaHeadDim = "mamba_head_dim"
        case mambaProjBias = "mamba_proj_bias"
        case ssmStateSize = "ssm_state_size"
        case convKernel = "conv_kernel"
        case nGroups = "n_groups"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case hybridOverridePattern = "hybrid_override_pattern"
        case layerNormEpsilon = "layer_norm_epsilon"
        case mlpBias = "mlp_bias"
        case useBias = "use_bias"
        case useConvBias = "use_conv_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeTheta = "rope_theta"
        case headDim = "head_dim"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case normTopkProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case timeStepLimitMin = "time_step_limit_min"
        case timeStepLimitMax = "time_step_limit_max"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "nemotron_h"
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        blockConfigurations = try container.decodeIfPresent(
            [BlockConfiguration].self, forKey: .blockConfigurations)
        moeLatentSize = try container.decodeIfPresent(Int.self, forKey: .moeLatentSize)
        numNextNPredictLayers =
            try container.decodeIfPresent(Int.self, forKey: .numNextNPredictLayers) ?? 0
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 0
        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mambaNumHeads = try container.decode(Int.self, forKey: .mambaNumHeads)
        mambaHeadDim = try container.decode(Int.self, forKey: .mambaHeadDim)
        mambaProjBias = try container.decodeIfPresent(Bool.self, forKey: .mambaProjBias) ?? false
        ssmStateSize = try container.decode(Int.self, forKey: .ssmStateSize)
        convKernel = try container.decode(Int.self, forKey: .convKernel)
        nGroups = try container.decode(Int.self, forKey: .nGroups)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        moeSharedExpertIntermediateSize = try container.decode(
            Int.self, forKey: .moeSharedExpertIntermediateSize)
        nRoutedExperts = try container.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 0
        nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        numExpertsPerTok = try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        layerNormEpsilon =
            try container.decodeIfPresent(Float.self, forKey: .layerNormEpsilon) ?? 1e-5
        mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        useBias = try container.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
        useConvBias = try container.decodeIfPresent(Bool.self, forKey: .useConvBias) ?? true
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try container.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        routedScalingFactor =
            try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0

        // Modern exports provide named blocks instead of the legacy character pattern.
        // Preserve explicit legacy overrides when a config also retains a block list.
        let extra = try decoder.container(keyedBy: ArchitectureKeys.self)
        if let blocks = blockConfigurations {
            hybridOverridePattern = try Self.pattern(blocks.map(\.blockType))
        } else if let pattern = try? container.decode(String.self, forKey: .hybridOverridePattern) {
            hybridOverridePattern = pattern
        } else if let pattern = try? container.decode([String].self, forKey: .hybridOverridePattern)
        {
            hybridOverridePattern = pattern.joined()
        } else if let blocks = try extra.decodeIfPresent([String].self, forKey: .layersBlockType) {
            hybridOverridePattern = try Self.pattern(blocks)
        } else {
            throw DecodingError.dataCorruptedError(
                forKey: .hybridOverridePattern, in: container,
                debugDescription: "Nemotron requires a block list or hybrid override pattern")
        }
        if numHiddenLayers == 0 { numHiddenLayers = hybridOverridePattern.count }
        guard numHiddenLayers == hybridOverridePattern.count,
            hybridOverridePattern.allSatisfy({ "M*-E".contains($0) })
        else {
            throw DecodingError.dataCorruptedError(
                forKey: .hybridOverridePattern, in: container,
                debugDescription: "Nemotron block pattern must match the layer count")
        }
        // Handle time_step_limit - can be array [min, max] or separate fields
        if let limits = try? container.decode([Float].self, forKey: .timeStepLimitMin) {
            // Actually this is time_step_limit as array
            timeStepLimitMin = limits[0]
            timeStepLimitMax = limits.count > 1 ? limits[1] : limits[0]
        } else {
            timeStepLimitMin =
                try container.decodeIfPresent(Float.self, forKey: .timeStepLimitMin) ?? 0.0
            timeStepLimitMax =
                try container.decodeIfPresent(Float.self, forKey: .timeStepLimitMax)
                ?? Float.infinity
        }
        for index in 0 ..< numHiddenLayers where Array(hybridOverridePattern)[index] == "E" {
            let layer = configuration(forLayer: index)
            guard layer.nRoutedExperts > 0, layer.moeIntermediateSize > 0,
                layer.numExpertsPerTok > 0, layer.numExpertsPerTok <= layer.nRoutedExperts
            else {
                throw DecodingError.dataCorruptedError(
                    forKey: .blockConfigurations, in: container,
                    debugDescription: "Invalid routed expert dimensions for layer \(index)")
            }
        }

    }

    private enum ArchitectureKeys: String, CodingKey {
        case layersBlockType = "layers_block_type"
    }

    private static func pattern(_ blocks: [String]) throws -> String {
        try String(
            blocks.map { block in
                switch block {
                case "mamba": return Character("M")
                case "attention": return Character("*")
                case "mlp": return Character("-")
                case "moe": return Character("E")
                default:
                    throw DecodingError.dataCorrupted(
                        .init(
                            codingPath: [],
                            debugDescription: "Unknown Nemotron block type: \(block)"))
                }
            })
    }

    public func configuration(forLayer index: Int) -> Self {
        guard let blockConfigurations else { return self }
        let block = blockConfigurations[index]
        var result = self
        result.blockConfigurations = nil
        result.nRoutedExperts = block.nRoutedExperts ?? nRoutedExperts
        result.moeIntermediateSize = block.moeIntermediateSize ?? moeIntermediateSize
        result.numExpertsPerTok = block.numExpertsPerTok ?? numExpertsPerTok
        return result
    }

    /// Memberwise initializer for testing
    public init(
        vocabSize: Int,
        hiddenSize: Int,
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        mambaNumHeads: Int,
        mambaHeadDim: Int,
        ssmStateSize: Int,
        convKernel: Int,
        nGroups: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        moeSharedExpertIntermediateSize: Int,
        nRoutedExperts: Int,
        numExpertsPerTok: Int,
        hybridOverridePattern: String,
        layerNormEpsilon: Float = 1e-5,
        attentionBias: Bool = false,
        mambaProjBias: Bool = false,
        mlpBias: Bool = false,
        useBias: Bool = false,
        useConvBias: Bool = true,
        tieWordEmbeddings: Bool = false,
        ropeTheta: Float = 10000.0,
        headDim: Int? = nil,
        nSharedExperts: Int? = nil,
        nGroup: Int = 1,
        topkGroup: Int = 1,
        normTopkProb: Bool = true,
        routedScalingFactor: Float = 1.0,
        timeStepLimitMin: Float = 0.0,
        timeStepLimitMax: Float = .infinity
    ) {
        self.modelType = "nemotron_h"
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.attentionBias = attentionBias
        self.mambaNumHeads = mambaNumHeads
        self.mambaHeadDim = mambaHeadDim
        self.mambaProjBias = mambaProjBias
        self.ssmStateSize = ssmStateSize
        self.convKernel = convKernel
        self.nGroups = nGroups
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.moeSharedExpertIntermediateSize = moeSharedExpertIntermediateSize
        self.nRoutedExperts = nRoutedExperts
        self.nSharedExperts = nSharedExperts
        self.numExpertsPerTok = numExpertsPerTok
        self.hybridOverridePattern = hybridOverridePattern
        self.layerNormEpsilon = layerNormEpsilon
        self.mlpBias = mlpBias
        self.useBias = useBias
        self.useConvBias = useConvBias
        self.tieWordEmbeddings = tieWordEmbeddings
        self.ropeTheta = ropeTheta
        self.headDim = headDim
        self.nGroup = nGroup
        self.topkGroup = topkGroup
        self.normTopkProb = normTopkProb
        self.routedScalingFactor = routedScalingFactor
        self.timeStepLimitMin = timeStepLimitMin
        self.timeStepLimitMax = timeStepLimitMax
    }
}
