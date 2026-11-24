import Foundation

public struct MistralLarge4Configuration: Decodable, Sendable {
    struct Text: Decodable, Sendable {
        let hiddenSize: Int
        let intermediateSize: Int
        let expertSize: Int
        let layers: Int
        let heads: Int
        let vocabularySize: Int
        let latentSize: Int
        let ropeSize: Int
        let contentSize: Int
        let valueSize: Int
        let experts: Int
        let sharedExperts: Int
        let routes: Int
        let denseLayers: Int
        let groups: Int
        let routeGroups: Int
        let routedScale: Float
        let eps: Float
        let indexHeads: Int
        let indexSize: Int
        let indexTopK: Int
        let softCap: Float
        let tiedEmbeddings: Bool
        let rotary: [Bool]
        let kvSources: [Int]
        let fullIndexers: [Bool]
        let rope: [String: StringOrNumber]

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case expertSize = "moe_intermediate_size"
            case layers = "num_hidden_layers"
            case heads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case vocabularySize = "vocab_size"
            case latentSize = "kv_lora_rank"
            case ropeSize = "qk_rope_head_dim"
            case contentSize = "qk_nope_head_dim"
            case valueSize = "v_head_dim"
            case eps = "rms_norm_eps"
            case experts = "n_routed_experts"
            case sharedExperts = "n_shared_experts"
            case routes = "num_experts_per_tok"
            case denseLayers = "first_k_dense_replace"
            case groups = "n_group"
            case routeGroups = "topk_group"
            case routedScale = "routed_scaling_factor"
            case indexHeads = "index_n_heads"
            case indexSize = "index_head_dim"
            case indexTopK = "index_topk"
            case softCap = "final_logit_softcapping"
            case tiedEmbeddings = "tie_word_embeddings"
            case rotary = "no_rope_layers"
            case rotaryInterval = "no_rope_layer_interval"
            case kvSources = "kv_source_layers"
            case indexerTypes = "indexer_types"
            case rope = "rope_parameters"
            case expertStart = "expert_start"
            case expertEnd = "expert_end"
        }

        init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            func int(_ key: CodingKeys, default fallback: Int) throws -> Int {
                try c.decodeIfPresent(Int.self, forKey: key) ?? fallback
            }
            hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
            intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
            layers = try c.decode(Int.self, forKey: .layers)
            heads = try c.decode(Int.self, forKey: .heads)
            vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
            latentSize = try c.decode(Int.self, forKey: .latentSize)
            ropeSize = try c.decode(Int.self, forKey: .ropeSize)
            contentSize = try c.decode(Int.self, forKey: .contentSize)
            valueSize = try c.decode(Int.self, forKey: .valueSize)
            experts = try int(.experts, default: 128)
            expertSize = try int(.expertSize, default: hiddenSize)
            sharedExperts = try int(.sharedExperts, default: 1)
            routes = try int(.routes, default: 4)
            denseLayers = try int(.denseLayers, default: 1)
            groups = try int(.groups, default: 1)
            routeGroups = try int(.routeGroups, default: 1)
            routedScale = try c.decodeIfPresent(Float.self, forKey: .routedScale) ?? 1
            eps = try c.decodeIfPresent(Float.self, forKey: .eps) ?? 1e-5
            indexHeads = try int(.indexHeads, default: 32)
            indexSize = try int(.indexSize, default: 128)
            indexTopK = try int(.indexTopK, default: 2048)
            softCap = try c.decodeIfPresent(Float.self, forKey: .softCap) ?? 30
            tiedEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tiedEmbeddings) ?? false
            rope = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .rope) ?? [:]
            let interval = try int(.rotaryInterval, default: 4)
            let kvHeads = try int(.kvHeads, default: heads)
            guard hiddenSize > 0, intermediateSize > 0, layers > 0, heads > 0, kvHeads == heads,
                vocabularySize > 0, latentSize > 0, ropeSize > 0, ropeSize.isMultiple(of: 2),
                contentSize > 0, valueSize > 0, indexHeads > 0, indexSize >= ropeSize,
                indexTopK > 0,
                interval > 0, (0 ... layers).contains(denseLayers),
                experts > 0, expertSize > 0, sharedExperts >= 0, groups > 0,
                experts.isMultiple(of: groups), (1 ... groups).contains(routeGroups),
                routes > 0, routes <= (experts / groups) * routeGroups,
                routedScale.isFinite, routedScale >= 0, eps.isFinite, eps > 0,
                softCap.isFinite, softCap >= 0
            else {
                throw ModelExecutionError.invalidOptions("Invalid Mistral Large 4 text dimensions")
            }
            let start = try int(.expertStart, default: 0)
            let end = try int(.expertEnd, default: experts)
            guard start == 0, end == experts else {
                throw ModelExecutionError.invalidOptions(
                    "Mistral Large 4 requires all experts on the local model")
            }
            if let raw = try c.decodeIfPresent([Int].self, forKey: .rotary) {
                guard raw.allSatisfy({ $0 == 0 || $0 == 1 }) else {
                    throw ModelExecutionError.invalidOptions(
                        "Invalid Mistral Large 4 rotary pattern")
                }
                // The checkpoint flags are 1 for rotary and 0 for NoPE.
                rotary = raw.map { $0 == 1 }
            } else {
                rotary = (0 ..< layers).map { ($0 + 1) % interval != 0 }
            }
            kvSources = try c.decodeIfPresent([Int].self, forKey: .kvSources) ?? Array(0 ..< layers)
            let indexers =
                try c.decodeIfPresent([String].self, forKey: .indexerTypes)
                ?? Array(repeating: "full", count: layers)
            guard rotary.count == layers, kvSources.count == layers, indexers.count == layers,
                indexers.first == "full", indexers.allSatisfy({ $0 == "full" || $0 == "shared" })
            else {
                throw ModelExecutionError.invalidOptions("Invalid Mistral Large 4 layer schedules")
            }
            for (layer, source) in kvSources.enumerated() {
                guard (0 ... layer).contains(source), kvSources[source] == source,
                    rotary[source] == rotary[layer]
                else {
                    throw ModelExecutionError.invalidOptions(
                        "Invalid Mistral Large 4 KV source for layer \(layer)")
                }
            }
            fullIndexers = indexers.map { $0 == "full" }
            guard ropeBase.isFinite, ropeBase > 1, ropeFactor.isFinite, ropeFactor > 0,
                ropeOriginal.isFinite, ropeOriginal > 0, ropeFast.isFinite, ropeFast > 0,
                ropeSlow.isFinite, ropeSlow > 0
            else {
                throw ModelExecutionError.invalidOptions(
                    "Invalid Mistral Large 4 LeRoPE parameters")
            }
        }

        var ropeBase: Float { rope["rope_theta"]?.asFloat() ?? 10_000 }
        var ropeFactor: Float { rope["factor"]?.asFloat() ?? 2048 }
        var ropeOriginal: Float { rope["original_max_position_embeddings"]?.asFloat() ?? 512 }
        var ropeFast: Float { rope["beta_fast"]?.asFloat() ?? 32 }
        var ropeSlow: Float { rope["beta_slow"]?.asFloat() ?? 1 }
    }

    let text: Text
    let vision: PixtralVisionConfiguration?
    let imageTokenIndex: Int
    let visionFeatureLayer: Int
    let spatialMergeSize: Int
    let projectorBias: Bool
    let fp8Block: [Int]

    enum CodingKeys: String, CodingKey {
        case text = "text_config"
        case vision = "vision_config"
        case imageTokenIndex = "image_token_index"
        case imageTokenId = "image_token_id"
        case visionFeatureLayer = "vision_feature_layer"
        case spatialMergeSize = "spatial_merge_size"
        case projectorBias = "multimodal_projector_bias"
        case quantization = "quantization_config"
    }

    private struct Quantization: Decodable {
        let weightBlockSize: [Int]?
        enum CodingKeys: String, CodingKey {
            case weightBlockSize = "weight_block_size"
        }
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        text = try c.decodeIfPresent(Text.self, forKey: .text) ?? Text(from: decoder)
        vision = try c.decodeIfPresent(PixtralVisionConfiguration.self, forKey: .vision)
        imageTokenIndex =
            try c.decodeIfPresent(Int.self, forKey: .imageTokenIndex)
            ?? c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 10
        visionFeatureLayer = try c.decodeIfPresent(Int.self, forKey: .visionFeatureLayer) ?? -1
        spatialMergeSize = try c.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? 2
        projectorBias = try c.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? false
        fp8Block =
            try c.decodeIfPresent(Quantization.self, forKey: .quantization)?.weightBlockSize
            ?? [128, 128]
        guard spatialMergeSize > 0, imageTokenIndex >= 0, fp8Block.count == 2,
            fp8Block.allSatisfy({ $0 > 0 })
        else {
            throw ModelExecutionError.invalidOptions(
                "Invalid Mistral Large 4 projector or FP8 format")
        }
    }

    static func normalizeNative(_ data: Data) throws -> Data? {
        guard let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let dsa = raw["dsa"] as? [String: Any], let moe = raw["moe"] as? [String: Any],
            raw["lerope"] as? Bool == true || raw["kv_sharing"] != nil,
            raw["dim"] != nil, let depth = raw["n_layers"] as? Int, depth > 0
        else { return nil }
        var text = raw
        text["model_type"] = "mistral_large4_text"
        for (from, to) in [
            ("dim", "hidden_size"), ("hidden_dim", "intermediate_size"),
            ("n_layers", "num_hidden_layers"), ("n_heads", "num_attention_heads"),
            ("n_kv_heads", "num_key_value_heads"), ("norm_eps", "rms_norm_eps"),
            ("softmax_tanh_gating", "final_logit_softcapping"),
            ("tied_embeddings", "tie_word_embeddings"),
        ] { text[to] = raw[from] }
        for (from, to) in [
            ("num_experts", "n_routed_experts"), ("num_shared_experts", "n_shared_experts"),
            ("expert_hidden_dim", "moe_intermediate_size"),
            ("num_experts_per_tok", "num_experts_per_tok"),
            ("num_expert_groups", "n_group"), ("num_expert_groups_per_tok", "topk_group"),
            ("routed_scale", "routed_scaling_factor"),
            ("first_k_dense_replace", "first_k_dense_replace"),
        ] { text[to] = moe[from] }
        for (from, to) in [
            ("topk", "index_topk"), ("head_dim", "index_head_dim"), ("n_heads", "index_n_heads"),
        ] {
            text[to] = dsa[from]
        }
        func pattern(_ string: String, choices: Set<String>) throws -> [String] {
            let values = string.components(separatedBy: ",")
            guard values.count <= depth, values.allSatisfy(choices.contains) else {
                throw ModelExecutionError.invalidOptions(
                    "Invalid Mistral Large 4 native layer pattern")
            }
            return (0 ..< depth).map { values[$0 % values.count] }
        }
        if let nope = raw["nope"] as? String {
            text["no_rope_layers"] = try pattern(nope, choices: ["rope", "nope"]).map {
                $0 == "rope" ? 1 : 0
            }
        }
        if let indexers = dsa["sharing"] as? String {
            text["indexer_types"] = try pattern(indexers, choices: ["full", "shared"])
        }
        if let sharing = raw["kv_sharing"] as? [String: Any],
            let indices = sharing["source_layers_str"] as? String
        {
            let parts = indices.components(separatedBy: ",")
            let sources = parts.compactMap(Int.init)
            guard sources.count == parts.count else {
                throw ModelExecutionError.invalidOptions(
                    "Invalid Mistral Large 4 native KV sources")
            }
            text["kv_source_layers"] = sources
        }
        let yarn = raw["yarn"] as? [String: Any] ?? [:]
        text["rope_parameters"] = [
            "rope_theta": raw["rope_theta"] ?? 10_000, "factor": yarn["factor"] ?? 1,
            "original_max_position_embeddings": yarn["original_max_position_embeddings"]
                ?? raw["max_position_embeddings"] ?? 512,
            "beta_fast": yarn["beta"] ?? 32, "beta_slow": yarn["alpha"] ?? 1,
        ]

        var result: [String: Any] = [
            "model_type": "mistral_large4_text", "text_config": text, "eos_token_id": 2,
        ]
        if var vision = raw["vision_encoder"] as? [String: Any], !vision.isEmpty {
            guard let width = vision["hidden_size"] as? Int,
                let heads = vision["num_attention_heads"] as? Int, heads > 0,
                width.isMultiple(of: heads)
            else {
                throw ModelExecutionError.invalidOptions(
                    "Invalid Mistral Large 4 native vision head size")
            }
            vision["model_type"] = "pixtral"
            vision["head_dim"] = vision["head_dim"] ?? width / heads
            if let theta = vision["rope_theta"] {
                vision["rope_parameters"] = ["rope_theta": theta]
            }
            result["model_type"] = "mistral_large4"
            result["vision_config"] = vision
            result["image_token_index"] = vision["image_token_id"]
            result["spatial_merge_size"] = vision["spatial_merge_size"]
            result["multimodal_projector_bias"] = vision["adapter_bias"]
        }
        if let quantization = raw["quantization"] as? [String: Any],
            let rows = quantization["block_size_out"] as? Int,
            let columns = quantization["block_size_in"] as? Int
        {
            result["quantization_config"] = ["weight_block_size": [rows, columns]]
        }
        return try JSONSerialization.data(withJSONObject: result)
    }
}
