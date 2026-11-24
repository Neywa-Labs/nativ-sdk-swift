import Foundation

public struct BailingHybridConfiguration: Decodable, Sendable {
    enum AttentionGate: String, Decodable, Sendable {
        case head = "head_wise"
        case element = "element_wise"
    }

    let hiddenSize: Int
    let intermediateSize: Int
    let vocabularySize: Int
    let layerCount: Int
    let heads: Int
    let headDim: Int
    let layerGroupSize: Int
    let convolutionSize: Int
    let fullRankKDA: Bool
    let lowerBound: Float?
    let queryRank: Int?
    let latentRank: Int
    let keyDim: Int
    let ropeDim: Int
    let valueDim: Int
    let qkvBias: Bool
    let attentionGate: AttentionGate?
    let normEpsilon: Float
    let ropeTheta: Float
    let ropeScaling: [String: StringOrNumber]?
    let ropeInterleaved: Bool
    let maxPositionEmbeddings: Int?
    let experts: Int
    let expertWidth: Int
    let sharedExperts: Int
    let sharedWidth: Int
    let expertsPerToken: Int
    let expertGroups: Int
    let groupsPerToken: Int
    let normalizeExperts: Bool
    let expertScale: Float
    let denseLayers: Int
    let expertClips: [Float]
    let sharedClips: [Float]
    let tiedEmbeddings: Bool
    let predictionLayers: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case vocabularySize = "vocab_size"
        case layerCount = "num_hidden_layers"
        case heads = "num_attention_heads"
        case headDim = "head_dim"
        case layerGroupSize = "layer_group_size"
        case convolutionSize = "short_conv_kernel_size"
        case fullRankKDA = "no_kda_lora"
        case safeGate = "kda_safe_gate"
        case lowerBound = "kda_lower_bound"
        case queryRank = "q_lora_rank"
        case latentRank = "kv_lora_rank"
        case keyDim = "qk_nope_head_dim"
        case ropeDim = "qk_rope_head_dim"
        case valueDim = "v_head_dim"
        case qkvBias = "use_qkv_bias"
        case attentionGate = "gated_attention_proj_granularity_type"
        case normEpsilon = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeInterleaved = "rope_interleave"
        case maxPositionEmbeddings = "max_position_embeddings"
        case experts = "num_experts"
        case expertWidth = "moe_intermediate_size"
        case sharedExperts = "num_shared_experts"
        case sharedWidth = "moe_shared_expert_intermediate_size"
        case expertsPerToken = "num_experts_per_tok"
        case expertGroups = "n_group"
        case groupsPerToken = "topk_group"
        case normalizeExperts = "norm_topk_prob"
        case expertScale = "routed_scaling_factor"
        case denseLayers = "first_k_dense_replace"
        case expertClips = "expert_swiglu_limit_list"
        case sharedClips = "share_expert_swiglu_limit_list"
        case tiedEmbeddings = "tie_word_embeddings"
        case predictionLayers = "num_nextn_predict_layers"
        case scoreFunction = "score_function"
        case activation = "hidden_act"
        case latentNoPE = "use_mla_nope"
        case linearSiLU = "linear_silu"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        layerCount = try c.decode(Int.self, forKey: .layerCount)
        heads = try c.decode(Int.self, forKey: .heads)
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        layerGroupSize = try c.decodeIfPresent(Int.self, forKey: .layerGroupSize) ?? 8
        convolutionSize = try c.decodeIfPresent(Int.self, forKey: .convolutionSize) ?? 4
        fullRankKDA = try c.decodeIfPresent(Bool.self, forKey: .fullRankKDA) ?? true
        let safeGate = try c.decodeIfPresent(Bool.self, forKey: .safeGate) ?? true
        lowerBound = safeGate ? try c.decodeIfPresent(Float.self, forKey: .lowerBound) ?? -5 : nil
        queryRank =
            c.contains(.queryRank) ? try c.decodeIfPresent(Int.self, forKey: .queryRank) : 1536
        latentRank = try c.decodeIfPresent(Int.self, forKey: .latentRank) ?? 512
        keyDim = try c.decodeIfPresent(Int.self, forKey: .keyDim) ?? 128
        ropeDim = try c.decodeIfPresent(Int.self, forKey: .ropeDim) ?? 64
        valueDim = try c.decodeIfPresent(Int.self, forKey: .valueDim) ?? 128
        qkvBias = try c.decodeIfPresent(Bool.self, forKey: .qkvBias) ?? false
        attentionGate =
            c.contains(.attentionGate)
            ? try c.decodeIfPresent(AttentionGate.self, forKey: .attentionGate) : .head
        normEpsilon = try c.decode(Float.self, forKey: .normEpsilon)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        ropeInterleaved = try c.decodeIfPresent(Bool.self, forKey: .ropeInterleaved) ?? true
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        experts = try c.decode(Int.self, forKey: .experts)
        expertWidth = try c.decode(Int.self, forKey: .expertWidth)
        sharedExperts = try c.decode(Int.self, forKey: .sharedExperts)
        let sharedWidth = try c.decodeIfPresent(Int.self, forKey: .sharedWidth) ?? 0
        self.sharedWidth = sharedWidth == 0 ? expertWidth : sharedWidth
        expertsPerToken = try c.decode(Int.self, forKey: .expertsPerToken)
        expertGroups = try c.decodeIfPresent(Int.self, forKey: .expertGroups) ?? 1
        groupsPerToken =
            try c.decodeIfPresent(Int.self, forKey: .groupsPerToken) ?? min(4, expertGroups)
        normalizeExperts = try c.decode(Bool.self, forKey: .normalizeExperts)
        expertScale = try c.decodeIfPresent(Float.self, forKey: .expertScale) ?? 1
        denseLayers = try c.decode(Int.self, forKey: .denseLayers)
        expertClips = try c.decodeIfPresent([Float].self, forKey: .expertClips) ?? []
        sharedClips = try c.decodeIfPresent([Float].self, forKey: .sharedClips) ?? []
        tiedEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tiedEmbeddings) ?? false
        predictionLayers = try c.decodeIfPresent(Int.self, forKey: .predictionLayers) ?? 0
        let score = try c.decodeIfPresent(String.self, forKey: .scoreFunction) ?? "sigmoid"
        let activation = try c.decodeIfPresent(String.self, forKey: .activation) ?? "silu"
        let noPE = try c.decodeIfPresent(Bool.self, forKey: .latentNoPE) ?? false
        let convolutionActivation = try c.decodeIfPresent(Bool.self, forKey: .linearSiLU) ?? true

        guard
            [
                hiddenSize, intermediateSize, vocabularySize, layerCount, heads, headDim,
                layerGroupSize, convolutionSize, latentRank, keyDim, ropeDim, valueDim,
                expertGroups,
            ]
            .allSatisfy({ $0 > 0 }),
            ropeDim.isMultiple(of: 2), queryRank.map({ $0 > 0 }) ?? true,
            maxPositionEmbeddings.map({ $0 > 0 }) ?? true,
            [normEpsilon, ropeTheta].allSatisfy({ $0.isFinite && $0 > 0 }),
            lowerBound.map({ $0.isFinite && $0 <= 0 }) ?? true,
            experts >= 0, sharedExperts >= 0, denseLayers >= 0, predictionLayers >= 0,
            expertScale.isFinite,
            (expertClips + sharedClips).allSatisfy({ $0.isFinite && $0 >= 0 }),
            score == "sigmoid", activation == "silu", !noPE, convolutionActivation
        else {
            throw ModelExecutionError.invalidOptions(
                "Invalid or unsupported Bailing hybrid configuration")
        }
        if experts > 0 {
            let grouped = expertGroups > 1 && groupsPerToken < expertGroups
            guard expertWidth > 0, self.sharedWidth > 0, experts.isMultiple(of: expertGroups),
                (1 ... expertGroups).contains(groupsPerToken),
                (1 ... experts).contains(expertsPerToken),
                !grouped
                    || (experts / expertGroups >= 2
                        && expertsPerToken <= (experts / expertGroups) * groupsPerToken)
            else {
                throw ModelExecutionError.invalidOptions("Invalid Bailing hybrid expert routing")
            }
        }
        try validateRoPEConfiguration(ropeScaling, context: "Bailing hybrid rope_scaling")
    }

    func isRecurrent(_ index: Int) -> Bool {
        index < layerCount / layerGroupSize * layerGroupSize
            && !(index + 1).isMultiple(of: layerGroupSize)
    }

    func usesExperts(_ index: Int) -> Bool { experts > 0 && index >= denseLayers }
}
