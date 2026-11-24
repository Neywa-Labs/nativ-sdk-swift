import Foundation

public struct Inkling2Configuration: Decodable, Sendable {
    struct Text: Decodable, Sendable {
        let width: Int
        let layers: Int
        let vocabulary: Int
        let unpaddedVocabulary: Int?
        let epsilon: Float
        let heads: Int
        let kvHeads: Int
        let headDimension: Int
        let slidingHeads: Int
        let slidingKVHeads: Int
        let slidingHeadDimension: Int
        let window: Int
        let layerTypes: [String]?
        let localLayers: [Int]?
        let relativeDimension: Int
        let relativeExtent: Int
        let logScalingFloor: Int?
        let logScalingAlpha: Float
        let kernel: Int
        let expertWidth: Int
        let experts: Int
        let topK: Int
        let routeScale: Float
        let qkClip: Float
        let relativeClip: Float
        let blockSize: Int
        let normalizeResidualValues: Bool
        let normalizeValues: Bool
        let gateCap: Float
        let upCap: Float
        let logitCap: Float

        enum CodingKeys: String, CodingKey {
            case width = "hidden_size"
            case layers = "num_hidden_layers"
            case vocabulary = "vocab_size"
            case unpaddedVocabulary = "unpadded_vocab_size"
            case epsilon = "rms_norm_eps"
            case heads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case headDimension = "head_dim"
            case slidingHeads = "swa_num_attention_heads"
            case slidingKVHeads = "swa_num_key_value_heads"
            case slidingHeadDimension = "swa_head_dim"
            case window = "sliding_window_size"
            case layerTypes = "layer_types"
            case localLayers = "local_layer_ids"
            case relativeDimension = "d_rel"
            case relativeExtent = "rel_extent"
            case logScalingFloor = "log_scaling_n_floor"
            case logScalingAlpha = "log_scaling_alpha"
            case kernel = "sconv_kernel_size"
            case expertWidth = "intermediate_size"
            case experts = "n_routed_experts"
            case topK = "num_experts_per_tok"
            case routeScale = "route_scale"
            case qkClip = "qk_l2_clip_threshold"
            case relativeClip = "rel_l2_clip_threshold"
            case blockSize = "n_layers_per_attn_res_block"
            case normalizeResidualValues = "attn_res_normalize_values"
            case normalizeValues = "use_value_norm_no_gamma"
            case gateCap = "mlp_gate_softcap"
            case upCap = "mlp_up_softcap"
            case logitCap = "final_logit_softcapping"
            case sharedExperts = "n_shared_experts"
            case queryBias = "q_bias"
            case outputBias = "o_bias"
            case tiedEmbeddings = "tie_word_embeddings"
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            width = try c.decode(Int.self, forKey: .width)
            layers = try c.decode(Int.self, forKey: .layers)
            vocabulary = try c.decode(Int.self, forKey: .vocabulary)
            unpaddedVocabulary = try c.decodeIfPresent(Int.self, forKey: .unpaddedVocabulary)
            epsilon = try c.decodeIfPresent(Float.self, forKey: .epsilon) ?? 1e-6
            heads = try c.decode(Int.self, forKey: .heads)
            kvHeads = try c.decode(Int.self, forKey: .kvHeads)
            headDimension = try c.decode(Int.self, forKey: .headDimension)
            slidingHeads = try c.decodeIfPresent(Int.self, forKey: .slidingHeads) ?? heads
            slidingKVHeads = try c.decodeIfPresent(Int.self, forKey: .slidingKVHeads) ?? kvHeads
            slidingHeadDimension =
                try c.decodeIfPresent(Int.self, forKey: .slidingHeadDimension) ?? headDimension
            window = try c.decodeIfPresent(Int.self, forKey: .window) ?? 1024
            layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes)
            localLayers = try c.decodeIfPresent([Int].self, forKey: .localLayers)
            relativeDimension = try c.decodeIfPresent(Int.self, forKey: .relativeDimension) ?? 16
            relativeExtent = try c.decodeIfPresent(Int.self, forKey: .relativeExtent) ?? 1024
            logScalingFloor =
                c.contains(.logScalingFloor)
                ? try c.decodeIfPresent(Int.self, forKey: .logScalingFloor) : 128_000
            logScalingAlpha = try c.decodeIfPresent(Float.self, forKey: .logScalingAlpha) ?? 0.1
            kernel = try c.decodeIfPresent(Int.self, forKey: .kernel) ?? 4
            expertWidth = try c.decode(Int.self, forKey: .expertWidth)
            experts = try c.decode(Int.self, forKey: .experts)
            topK = try c.decode(Int.self, forKey: .topK)
            routeScale = try c.decodeIfPresent(Float.self, forKey: .routeScale) ?? 8
            qkClip = try c.decodeIfPresent(Float.self, forKey: .qkClip) ?? 22.627417
            relativeClip = try c.decodeIfPresent(Float.self, forKey: .relativeClip) ?? 8
            blockSize = try c.decodeIfPresent(Int.self, forKey: .blockSize) ?? 6
            normalizeResidualValues =
                try c.decodeIfPresent(Bool.self, forKey: .normalizeResidualValues) ?? true
            normalizeValues = try c.decodeIfPresent(Bool.self, forKey: .normalizeValues) ?? true
            gateCap = try c.decodeIfPresent(Float.self, forKey: .gateCap) ?? 4
            upCap = try c.decodeIfPresent(Float.self, forKey: .upCap) ?? 25
            logitCap = try c.decodeIfPresent(Float.self, forKey: .logitCap) ?? 30
            let shared = try c.decodeIfPresent(Int.self, forKey: .sharedExperts) ?? 0
            let queryBias = try c.decodeIfPresent(Bool.self, forKey: .queryBias) ?? false
            let outputBias = try c.decodeIfPresent(Bool.self, forKey: .outputBias) ?? false
            let tied = try c.decodeIfPresent(Bool.self, forKey: .tiedEmbeddings) ?? false

            guard
                [
                    width, layers, vocabulary, heads, kvHeads, headDimension, slidingHeads,
                    slidingKVHeads, slidingHeadDimension, window, relativeDimension, relativeExtent,
                    kernel, expertWidth, experts, topK, blockSize, logScalingFloor ?? 1,
                ].allSatisfy({ $0 > 0 }),
                heads.isMultiple(of: kvHeads), slidingHeads.isMultiple(of: slidingKVHeads),
                topK <= experts, (1 ... vocabulary).contains(unpaddedVocabulary ?? vocabulary),
                layerTypes.map({ types in
                    types.count == layers
                        && types.allSatisfy {
                            $0 == "hybrid_sliding" || $0 == "hybrid_full"
                        }
                }) ?? true,
                localLayers?.allSatisfy({ (0 ..< layers).contains($0) }) ?? true,
                [epsilon, routeScale, qkClip, relativeClip, gateCap, upCap].allSatisfy({
                    $0.isFinite && $0 > 0
                }), logitCap.isFinite, logitCap >= 0, logScalingAlpha.isFinite,
                shared == 0, !queryBias, !outputBias, !tied
            else {
                throw ModelExecutionError.invalidOptions("Invalid Inkling 2 text configuration")
            }
        }

        func isSliding(_ layer: Int) -> Bool {
            if let layerTypes { return layerTypes[layer] == "hybrid_sliding" }
            if let localLayers { return localLayers.contains(layer) }
            return (layer + 1) % 6 != 0
        }
    }

    struct Vision: Decodable, Sendable {
        let patch: Int
        let merge: Int
        let projectionWidth: Int
        let projectorBias: Bool
        let width: Int
        let layers: Int
        let heads: Int
        let mlpRatio: Float
        let positionGrid: Int
        let epsilon: Float
        let theta: Float

        enum CodingKeys: String, CodingKey {
            case patch = "native_patch_size"
            case merge = "adapter_merge_size"
            case projectionWidth = "projection_hidden_dim"
            case projectorBias = "projector_bias"
            case width = "hidden_size"
            case layers = "num_hidden_layers"
            case heads = "num_attention_heads"
            case mlpRatio = "mlp_ratio"
            case positionGrid = "posemb_grid_size"
            case epsilon = "layer_norm_eps"
            case theta = "rope_theta"
            case channels = "num_channels"
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            patch = try c.decodeIfPresent(Int.self, forKey: .patch) ?? 20
            merge = try c.decodeIfPresent(Int.self, forKey: .merge) ?? 2
            projectionWidth = try c.decode(Int.self, forKey: .projectionWidth)
            projectorBias = try c.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? false
            width = try c.decode(Int.self, forKey: .width)
            layers = try c.decode(Int.self, forKey: .layers)
            heads = try c.decode(Int.self, forKey: .heads)
            mlpRatio = try c.decodeIfPresent(Float.self, forKey: .mlpRatio) ?? 4
            positionGrid = try c.decodeIfPresent(Int.self, forKey: .positionGrid) ?? 32
            epsilon = try c.decodeIfPresent(Float.self, forKey: .epsilon) ?? 1e-5
            theta = try c.decodeIfPresent(Float.self, forKey: .theta) ?? 10000
            let channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            guard
                [patch, merge, projectionWidth, width, layers, heads, positionGrid]
                    .allSatisfy({ $0 > 0 }),
                width.isMultiple(of: heads), (width / heads).isMultiple(of: 4), channels == 3,
                [mlpRatio, epsilon, theta].allSatisfy({ $0.isFinite && $0 > 0 })
            else {
                throw ModelExecutionError.invalidOptions("Invalid Inkling 2 vision configuration")
            }
        }

        var tile: Int { patch * merge }
    }

    let text: Text
    let vision: Vision?
    let imageToken: Int

    enum CodingKeys: String, CodingKey {
        case text = "text_config"
        case vision = "vision_config"
        case imageToken = "image_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        text = try c.decode(Text.self, forKey: .text)
        vision = try c.decodeIfPresent(Vision.self, forKey: .vision)
        imageToken = try c.decodeIfPresent(Int.self, forKey: .imageToken) ?? 200054
    }
}
