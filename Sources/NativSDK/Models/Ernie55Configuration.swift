import Foundation

public struct Ernie55Configuration: Decodable, Sendable {
    let hidden: Int
    let vocabulary: Int
    let denseWidth: Int
    let layers: Int
    let denseLayers: Int
    let compression: [Int]
    let heads: Int
    let headDim: Int
    let queryRank: Int
    let ropeDim: Int
    let outputGroups: Int
    let outputRank: Int
    let window: Int
    let mlaHeads: Int
    let mlaQueryRank: Int
    let mlaLatent: Int
    let mlaNope: Int
    let mlaRope: Int
    let mlaValue: Int
    let indexHeads: Int
    let indexDim: Int
    let indexTopK: Int
    let maxPositions: Int
    let theta: Float
    let compressedTheta: Float
    let ropeScaling: [String: StringOrNumber]?
    let rmsEps: Float
    let hcStreams: Int
    let sinkhornIterations: Int
    let hcEps: Float
    let expertWidth: Int
    let expertLatent: Int
    let experts: Int
    let sharedExperts: Int
    let topK: Int
    let routeScale: Float
    let normalizeRouting: Bool
    let gateCap: Float
    let upCap: Float
    let virtualRank: Int

    enum CodingKeys: String, CodingKey {
        case hidden = "hidden_size"
        case vocabulary = "vocab_size"
        case denseWidth = "intermediate_size"
        case layers = "num_hidden_layers"
        case denseLayers = "first_k_dense_replace"
        case compression = "compress_ratios"
        case heads = "num_attention_heads"
        case headDim = "head_dim"
        case queryRank = "q_lora_rank"
        case ropeDim = "qk_rope_head_dim"
        case outputGroups = "o_groups"
        case outputRank = "o_lora_rank"
        case window = "sliding_window"
        case mlaHeads = "hybrid_mla_num_attention_heads"
        case mlaQueryRank = "hybrid_mla_q_lora_rank"
        case mlaLatent = "hybrid_mla_kv_lora_rank"
        case mlaNope = "hybrid_mla_qk_nope_head_dim"
        case mlaRope = "hybrid_mla_qk_rope_head_dim"
        case mlaValue = "hybrid_mla_v_head_dim"
        case indexHeads = "index_n_heads"
        case indexDim = "index_head_dim"
        case indexTopK = "index_topk"
        case maxPositions = "max_position_embeddings"
        case theta = "rope_theta"
        case compressedTheta = "compress_rope_theta"
        case ropeScaling = "rope_scaling"
        case rmsEps = "rms_norm_eps"
        case hcStreams = "hc_mult"
        case sinkhornIterations = "hc_sinkhorn_iters"
        case hcEps = "hc_eps"
        case expertWidth = "moe_intermediate_size"
        case expertLatent = "moe_latent_size"
        case experts = "n_routed_experts"
        case sharedExperts = "n_shared_experts"
        case topK = "num_experts_per_tok"
        case routeScale = "routed_scaling_factor"
        case normalizeRouting = "norm_topk_prob"
        case gateCap = "activation_situ_beta"
        case upCap = "activation_situ_linear_beta"
        case virtualRank = "vha_rank"
        case activation = "hidden_act"
        case attentionBias = "attention_bias"
        case tied = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func read<T: Decodable>(_ key: CodingKeys, _ fallback: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: key) ?? fallback
        }
        hidden = try read(.hidden, 4096)
        vocabulary = try read(.vocabulary, 201216)
        denseWidth = try read(.denseWidth, 14336)
        layers = try read(.layers, 43)
        denseLayers = try read(.denseLayers, 1)
        let ratios: [Int] = try read(.compression, [])
        compression =
            ratios.isEmpty
            ? (0 ..< max(0, layers)).map { $0 > 0 && $0.isMultiple(of: 7) ? -2 : 128 }
            : Array(ratios.prefix(max(0, layers)))
        heads = try read(.heads, 64)
        headDim = try read(.headDim, 512)
        queryRank = try read(.queryRank, 1024)
        ropeDim = try read(.ropeDim, 64)
        outputGroups = try read(.outputGroups, 8)
        outputRank = try read(.outputRank, 1024)
        window = try read(.window, 128)
        mlaHeads = try read(.mlaHeads, 64)
        mlaQueryRank = try read(.mlaQueryRank, 1024)
        mlaLatent = try read(.mlaLatent, 512)
        mlaNope = try read(.mlaNope, 192)
        mlaRope = try read(.mlaRope, 64)
        mlaValue = try read(.mlaValue, 256)
        indexHeads = try read(.indexHeads, 64)
        indexDim = try read(.indexDim, 128)
        indexTopK = try read(.indexTopK, 2048)
        maxPositions = try read(.maxPositions, 1_048_576)
        theta = try read(.theta, 8_000_000)
        compressedTheta = try read(.compressedTheta, 160_000)
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        rmsEps = try read(.rmsEps, 1e-6)
        hcStreams = try read(.hcStreams, 4)
        sinkhornIterations = try read(.sinkhornIterations, 20)
        hcEps = try read(.hcEps, 1e-6)
        expertWidth = try read(.expertWidth, 2048)
        expertLatent = try read(.expertLatent, 2048)
        experts = try read(.experts, 512)
        sharedExperts = try read(.sharedExperts, 1)
        topK = try read(.topK, 10)
        routeScale = try read(.routeScale, 3)
        normalizeRouting = try read(.normalizeRouting, true)
        gateCap = try read(.gateCap, 4)
        upCap = try read(.upCap, 25)
        virtualRank = try read(.virtualRank, 16)

        guard
            [
                hidden, vocabulary, denseWidth, layers, heads, headDim, queryRank, ropeDim,
                outputGroups, outputRank, window, mlaHeads, mlaQueryRank, mlaLatent, mlaNope,
                mlaRope, mlaValue, indexHeads, indexDim, indexTopK, maxPositions, hcStreams,
                sinkhornIterations, expertWidth, expertLatent, experts, sharedExperts, topK,
                virtualRank,
            ].allSatisfy({ $0 > 0 }),
            (0 ... layers).contains(denseLayers), topK <= experts,
            heads.isMultiple(of: outputGroups),
            headDim.isMultiple(of: 2), ropeDim.isMultiple(of: 2), ropeDim <= headDim,
            mlaRope.isMultiple(of: 2), mlaRope <= indexDim,
            compression.count == layers,
            compression.allSatisfy({ $0 == -2 || ($0 > 0 && $0 != 4) }),
            [theta, compressedTheta, rmsEps, hcEps, routeScale, gateCap, upCap]
                .allSatisfy({ $0.isFinite && $0 > 0 }),
            try read(.activation, "situ") == "situ", try !read(.attentionBias, false),
            try !read(.tied, false)
        else {
            throw ModelExecutionError.invalidOptions(
                "Invalid or unsupported ERNIE 5.5 configuration")
        }
    }
}
