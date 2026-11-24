import Foundation

public struct Raon1MiniConfiguration: Decodable, Sendable {
    struct Delta: Decodable, Sendable {
        let heads: Int?
        let headDim: Int?
        let kernelSize: Int
        let lowerBound: Float?
        let fullRankGate: Bool
        let recurrentLayers: [Int]
        let latentLayers: [Int]

        enum CodingKeys: String, CodingKey {
            case heads = "num_heads"
            case headDim = "head_dim"
            case kernelSize = "conv_kernel_size"
            case lowerBound = "gate_lower_bound"
            case fullRankGate = "full_rank_gate"
            case recurrentLayers = "rda_layers"
            case latentLayers = "mla_layers"
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            heads = try c.decodeIfPresent(Int.self, forKey: .heads)
            headDim = try c.decodeIfPresent(Int.self, forKey: .headDim)
            kernelSize = try c.decodeIfPresent(Int.self, forKey: .kernelSize) ?? 4
            lowerBound =
                c.contains(.lowerBound)
                ? try c.decodeIfPresent(Float.self, forKey: .lowerBound) : -5
            fullRankGate = try c.decodeIfPresent(Bool.self, forKey: .fullRankGate) ?? true
            recurrentLayers = try c.decodeIfPresent([Int].self, forKey: .recurrentLayers) ?? []
            latentLayers = try c.decodeIfPresent([Int].self, forKey: .latentLayers) ?? []
        }

        static let preview = Delta()
        private init() {
            heads = nil
            headDim = nil
            kernelSize = 4
            lowerBound = -5
            fullRankGate = true
            recurrentLayers = (1 ... 23).filter { !$0.isMultiple(of: 4) }
            latentLayers = [4, 8, 12, 16, 20, 24, 25]
        }
    }

    private struct SourceQuantization: Decodable {
        let method: String?
        let blockSize: [Int]?
        enum CodingKeys: String, CodingKey {
            case method = "quant_method"
            case blockSize = "weight_block_size"
        }
    }

    let hiddenSize: Int
    let vocabularySize: Int
    let layerCount: Int
    let heads: Int
    let headDim: Int
    let intermediateSize: Int
    let normEpsilon: Float
    let latentNormEpsilon: Float
    let selectorEpsilon: Float
    let streams: Int
    let postFactor: Float
    let gateBeta: Float
    let valueBeta: Float
    let experts: Int
    let expertsPerToken: Int
    let expertWidth: Int
    let sharedExperts: Int
    let denseLayers: Int
    let expertFrequency: Int
    let renormalize: Bool
    let expertScale: Float
    let queryRank: Int
    let latentRank: Int
    let keyHeadDim: Int
    let extraKeyDim: Int
    let valueHeadDim: Int
    let latentOutputGate: Bool
    let delta: Delta
    let decayRank: Int
    let extraPredictionLayers: Int
    let tiedEmbeddings: Bool
    let usesSourceMXFP8: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case vocabularySize = "vocab_size"
        case layerCount = "num_hidden_layers"
        case heads = "num_attention_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case normEpsilon = "rms_norm_eps"
        case latentNormEpsilon = "mla_latent_norm_eps"
        case selectorEpsilon = "raon_mhc_selector_eps"
        case streams = "raon_mhc_streams"
        case postFactor = "raon_mhc_post_factor"
        case activation = "hidden_act"
        case gateBeta = "activation_situ_beta"
        case valueBeta = "activation_situ_linear_beta"
        case experts = "num_experts"
        case expertsPerToken = "num_experts_per_token"
        case expertWidth = "moe_intermediate_size"
        case sharedExperts = "num_shared_experts"
        case denseLayers = "first_k_dense_replace"
        case expertFrequency = "moe_layer_freq"
        case router = "moe_router_activation_func"
        case renormalize = "moe_renormalize"
        case expertScale = "routed_scaling_factor"
        case groupedTopK = "use_grouped_topk"
        case routedExpertWidth = "routed_expert_hidden_size"
        case latentMoENorm = "latent_moe_use_norm"
        case queryRank = "q_lora_rank"
        case latentRank = "kv_lora_rank"
        case keyHeadDim = "qk_nope_head_dim"
        case extraKeyDim = "qk_rope_head_dim"
        case valueHeadDim = "v_head_dim"
        case noPositionEncoding = "mla_use_nope"
        case latentOutputGate = "mla_use_output_gate"
        case delta = "rda_config"
        case decayRank = "rda_decay_rank"
        case extraPredictionLayers = "num_nextn_predict_layers"
        case attentionResidualBlock = "attn_res_block_size"
        case tiedEmbeddings = "tie_word_embeddings"
        case sourceQuantization = "quantization_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2560
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 259840
        layerCount = try c.decodeIfPresent(Int.self, forKey: .layerCount) ?? 25
        heads = try c.decodeIfPresent(Int.self, forKey: .heads) ?? 40
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 12032
        normEpsilon = try c.decodeIfPresent(Float.self, forKey: .normEpsilon) ?? 1e-5
        latentNormEpsilon = try c.decodeIfPresent(Float.self, forKey: .latentNormEpsilon) ?? 1e-6
        selectorEpsilon = try c.decodeIfPresent(Float.self, forKey: .selectorEpsilon) ?? 1e-5
        streams = try c.decodeIfPresent(Int.self, forKey: .streams) ?? 4
        postFactor = try c.decodeIfPresent(Float.self, forKey: .postFactor) ?? 2
        gateBeta = try c.decodeIfPresent(Float.self, forKey: .gateBeta) ?? 4
        valueBeta = try c.decodeIfPresent(Float.self, forKey: .valueBeta) ?? 25
        experts =
            c.contains(.experts) ? try c.decodeIfPresent(Int.self, forKey: .experts) ?? 0 : 128
        expertsPerToken = try c.decodeIfPresent(Int.self, forKey: .expertsPerToken) ?? 6
        expertWidth = try c.decodeIfPresent(Int.self, forKey: .expertWidth) ?? 1088
        sharedExperts = try c.decodeIfPresent(Int.self, forKey: .sharedExperts) ?? 1
        denseLayers = try c.decodeIfPresent(Int.self, forKey: .denseLayers) ?? 1
        expertFrequency = try c.decodeIfPresent(Int.self, forKey: .expertFrequency) ?? 1
        renormalize = try c.decodeIfPresent(Bool.self, forKey: .renormalize) ?? true
        expertScale = try c.decodeIfPresent(Float.self, forKey: .expertScale) ?? 1
        queryRank = try c.decodeIfPresent(Int.self, forKey: .queryRank) ?? 576
        latentRank = try c.decodeIfPresent(Int.self, forKey: .latentRank) ?? 512
        keyHeadDim = try c.decodeIfPresent(Int.self, forKey: .keyHeadDim) ?? 128
        extraKeyDim = try c.decodeIfPresent(Int.self, forKey: .extraKeyDim) ?? 64
        valueHeadDim = try c.decodeIfPresent(Int.self, forKey: .valueHeadDim) ?? 128
        latentOutputGate = try c.decodeIfPresent(Bool.self, forKey: .latentOutputGate) ?? true
        delta = try c.decodeIfPresent(Delta.self, forKey: .delta) ?? .preview
        decayRank = try c.decodeIfPresent(Int.self, forKey: .decayRank) ?? 128
        extraPredictionLayers = try c.decodeIfPresent(Int.self, forKey: .extraPredictionLayers) ?? 0
        tiedEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tiedEmbeddings) ?? false
        let source = try c.decodeIfPresent(SourceQuantization.self, forKey: .sourceQuantization)
        usesSourceMXFP8 = source?.method == "mxfp8"

        let activation = try c.decodeIfPresent(String.self, forKey: .activation) ?? "situ"
        let router = try c.decodeIfPresent(String.self, forKey: .router) ?? "sigmoid"
        let grouped = try c.decodeIfPresent(Bool.self, forKey: .groupedTopK) ?? false
        let routedWidth = try c.decodeIfPresent(Int.self, forKey: .routedExpertWidth)
        let latentNorm = try c.decodeIfPresent(Bool.self, forKey: .latentMoENorm) ?? false
        let nope = try c.decodeIfPresent(Bool.self, forKey: .noPositionEncoding) ?? true
        let residualBlock = try c.decodeIfPresent(Int.self, forKey: .attentionResidualBlock)
        guard activation == "situ", router == "sigmoid", !grouped, routedWidth == nil, !latentNorm,
            nope, residualBlock == nil, delta.fullRankGate,
            source?.method == nil || usesSourceMXFP8,
            source?.blockSize == nil || source?.blockSize == [1, 32],
            [
                hiddenSize, vocabularySize, layerCount, heads, headDim, intermediateSize, queryRank,
                latentRank, keyHeadDim, extraKeyDim, valueHeadDim, recurrentHeads, recurrentHeadDim,
                delta.kernelSize, decayRank, expertWidth, expertFrequency,
            ].allSatisfy({ $0 > 0 }),
            streams >= 2, experts >= 0, sharedExperts >= 0, denseLayers >= 0,
            extraPredictionLayers >= 0,
            experts == 0 || (expertsPerToken > 0 && expertsPerToken <= experts),
            [normEpsilon, latentNormEpsilon, selectorEpsilon, gateBeta, valueBeta, postFactor]
                .allSatisfy({ $0.isFinite && $0 > 0 }),
            delta.lowerBound.map({ $0.isFinite && $0 <= 0 }) ?? true, expertScale.isFinite,
            (delta.recurrentLayers + delta.latentLayers).sorted() == Array(1 ... layerCount)
        else {
            throw ModelExecutionError.invalidOptions(
                "Invalid or unsupported Raon-1 Mini configuration")
        }
    }

    var recurrentHeads: Int { delta.heads ?? heads }
    var recurrentHeadDim: Int { delta.headDim ?? headDim }
    func usesExperts(at index: Int) -> Bool {
        experts > 0 && index >= denseLayers && index.isMultiple(of: expertFrequency)
    }
}
