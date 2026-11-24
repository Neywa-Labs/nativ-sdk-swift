import Foundation

public struct MuseSparkConfiguration: Decodable, Sendable {
    struct Text: Decodable, Sendable {
        let hiddenSize: Int
        let heads: Int
        let kvHeads: Int
        let headDimension: Int
        let layers: Int
        let vocabularySize: Int
        let rmsEpsilon: Float
        let postEpsilon: Float
        let ropeParameters: [String: StringOrNumber]
        let layerTheta: [Float]
        let layerTypes: [String]
        let slidingWindow: Int
        let logitCap: Float
        let outputMultiplier: Float
        let expertCount: Int
        let topK: Int
        let expertWidth: Int
        let expertIntermediate: Int
        let queryScale: Float
        let routeScale: Float
        let gateTemperature: Float

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case heads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case headDimension = "head_dim"
            case layers = "num_hidden_layers"
            case vocabularySize = "vocab_size"
            case rmsEpsilon = "rms_norm_eps"
            case postEpsilon = "post_norm_eps"
            case ropeParameters = "rope_parameters"
            case layerTheta = "layer_rope_theta"
            case layerTypes = "layer_types"
            case slidingWindow = "sliding_window"
            case logitCap = "final_logit_softcapping"
            case outputMultiplier = "output_multiplier"
            case expertCount = "num_local_experts"
            case topK = "num_experts_per_tok"
            case expertWidth = "moe_hidden_size"
            case expertIntermediate = "moe_intermediate_size"
            case queryScale = "qk_scale_factor"
            case routeScale = "routed_scaling_factor"
            case gateTemperature = "residual_gate_temperature"
            case sharedExperts = "num_shared_experts"
            case mlpLayerTypes = "mlp_layer_types"
            case attentionBias = "attention_bias"
            case tiedEmbeddings = "tie_word_embeddings"
            case activation = "hidden_act"
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
            heads = try c.decode(Int.self, forKey: .heads)
            kvHeads = try c.decode(Int.self, forKey: .kvHeads)
            headDimension = try c.decode(Int.self, forKey: .headDimension)
            layers = try c.decode(Int.self, forKey: .layers)
            vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
            rmsEpsilon = try c.decode(Float.self, forKey: .rmsEpsilon)
            postEpsilon = try c.decode(Float.self, forKey: .postEpsilon)
            ropeParameters = try c.decode([String: StringOrNumber].self, forKey: .ropeParameters)
            layerTheta = Array(
                try c.decode([Float].self, forKey: .layerTheta).prefix(max(0, layers)))
            layerTypes = Array(
                try c.decode([String].self, forKey: .layerTypes).prefix(max(0, layers)))
            slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
            logitCap = try c.decode(Float.self, forKey: .logitCap)
            outputMultiplier = try c.decode(Float.self, forKey: .outputMultiplier)
            expertCount = try c.decode(Int.self, forKey: .expertCount)
            topK = try c.decode(Int.self, forKey: .topK)
            expertWidth = try c.decode(Int.self, forKey: .expertWidth)
            expertIntermediate = try c.decode(Int.self, forKey: .expertIntermediate)
            queryScale = try c.decodeIfPresent(Float.self, forKey: .queryScale) ?? 5.4730064863875
            routeScale = try c.decodeIfPresent(Float.self, forKey: .routeScale) ?? 1
            gateTemperature = try c.decodeIfPresent(Float.self, forKey: .gateTemperature) ?? 0.3
            let shared = try c.decodeIfPresent(Int.self, forKey: .sharedExperts) ?? 0
            let mlps = try c.decodeIfPresent([String].self, forKey: .mlpLayerTypes) ?? []
            let biases = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            let tied = try c.decodeIfPresent(Bool.self, forKey: .tiedEmbeddings) ?? false
            let activation = try c.decodeIfPresent(String.self, forKey: .activation) ?? "silu"

            guard
                [
                    hiddenSize, heads, kvHeads, headDimension, layers, vocabularySize,
                    slidingWindow,
                    expertCount, expertWidth, expertIntermediate, topK,
                ].allSatisfy({ $0 > 0 }),
                heads.isMultiple(of: kvHeads), headDimension.isMultiple(of: 2),
                topK <= expertCount, layerTheta.count == layers, layerTypes.count == layers,
                validPositive(ropeParameters["rope_theta"]?.asFloat() ?? 0),
                layerTheta.allSatisfy({ $0 >= 0 && $0.isFinite }),
                layerTypes.allSatisfy({ $0 == "full_attention" || $0 == "sliding_attention" }),
                [
                    rmsEpsilon, postEpsilon, outputMultiplier, queryScale, routeScale,
                    gateTemperature,
                ]
                .allSatisfy(validPositive),
                logitCap >= 0, logitCap.isFinite, shared == 0, !biases, !tied,
                activation == "silu", mlps.allSatisfy({ $0 == "sparse" })
            else {
                throw ModelExecutionError.invalidOptions(
                    "Invalid or unsupported Muse Spark text configuration")
            }
        }

        var ropeBase: Float { ropeParameters["rope_theta"]!.asFloat()! }
        func isSliding(_ index: Int) -> Bool { layerTypes[index] == "sliding_attention" }
    }

    struct Vision: Decodable, Sendable {
        let geometry: MuseGlimmerVisionConfiguration
        let useIJPositions: Bool

        enum CodingKeys: String, CodingKey {
            case useIJPositions = "use_ij_pos_encoding"
            case channels = "num_channels"
        }

        init(from decoder: Decoder) throws {
            geometry = try MuseGlimmerVisionConfiguration(from: decoder)
            let c = try decoder.container(keyedBy: CodingKeys.self)
            useIJPositions = try c.decodeIfPresent(Bool.self, forKey: .useIJPositions) ?? false
            let channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            let g = geometry
            guard channels == 3,
                [
                    g.hiddenSize, g.intermediateSize, g.attentionHeads, g.hiddenLayers,
                    g.patchSize, g.patchTemporal, g.mergeSize, g.positionEmbeddingHeight,
                ].allSatisfy({ $0 > 0 }),
                g.hiddenSize.isMultiple(of: g.attentionHeads),
                (g.hiddenSize / g.attentionHeads).isMultiple(of: 4),
                g.positionEmbeddingHeight == g.positionEmbeddingWidth,
                g.layerTypes.count >= g.hiddenLayers,
                g.layerTypes.prefix(g.hiddenLayers).allSatisfy({
                    $0 == "full_attention" || $0 == "window_attention"
                }),
                validPositive(g.layerNormEps), validPositive(g.ropeTheta)
            else {
                throw ModelExecutionError.invalidOptions(
                    "Invalid or unsupported Muse Spark vision configuration")
            }
        }
    }

    struct Audio: Decodable, Sendable {
        let patchSizes: [Int]
        let hiddenSizes: [Int]
        let depths: [Int]
        let mlpRatio: Float
        let kernelSize: Int
        let epsilon: Float
        let outputSize: Int
        let samplesPerToken: Int

        enum CodingKeys: String, CodingKey {
            case patchSizes = "patch_sizes"
            case hiddenSizes = "hidden_sizes"
            case depths
            case layers = "num_hidden_layers"
            case mlpRatio = "mlp_ratio"
            case kernelSize = "conv_kernel_size"
            case epsilon = "rms_norm_eps"
            case outputSize = "out_hidden_size"
            case activation = "hidden_act"
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            patchSizes = try c.decodeIfPresent([Int].self, forKey: .patchSizes) ?? [16, 6, 5, 4]
            hiddenSizes =
                try c.decodeIfPresent([Int].self, forKey: .hiddenSizes) ?? [256, 512, 1024, 2048]
            depths = try c.decodeIfPresent([Int].self, forKey: .depths) ?? [2, 2, 2, 2]
            let layers = try c.decodeIfPresent(Int.self, forKey: .layers) ?? 4
            mlpRatio = try c.decodeIfPresent(Float.self, forKey: .mlpRatio) ?? 4
            kernelSize = try c.decodeIfPresent(Int.self, forKey: .kernelSize) ?? 7
            epsilon = try c.decodeIfPresent(Float.self, forKey: .epsilon) ?? 1e-6
            outputSize = try c.decodeIfPresent(Int.self, forKey: .outputSize) ?? 8192
            let activation = try c.decodeIfPresent(String.self, forKey: .activation) ?? "gelu"
            let stride = patchSizes.reduce(1) { product, size in
                let (value, overflow) = product.multipliedReportingOverflow(by: size)
                return overflow ? 0 : value
            }
            samplesPerToken = stride
            guard layers > 0, patchSizes.count == layers, hiddenSizes.count == layers,
                depths.count == layers, patchSizes.allSatisfy({ $0 > 0 }),
                depths.allSatisfy({ $0 >= 0 }), hiddenSizes.allSatisfy({ $0 > 0 }),
                kernelSize > 0, stride > 0, outputSize > 0, activation == "gelu",
                validPositive(mlpRatio), validPositive(epsilon),
                hiddenSizes.allSatisfy({ Float($0) * mlpRatio >= 1 })
            else {
                throw ModelExecutionError.invalidOptions(
                    "Invalid or unsupported Muse Spark audio configuration")
            }
        }
    }

    let text: Text
    let vision: Vision?
    let audio: Audio?
    let imageToken: Int
    let videoToken: Int
    let audioToken: Int
    let visionOutputSize: Int
    let projectorSize: Int

    enum CodingKeys: String, CodingKey {
        case text = "text_config"
        case vision = "vision_config"
        case audio = "audio_config"
        case imageToken = "image_token_id"
        case videoToken = "video_token_id"
        case audioToken = "audio_token_id"
        case visionOutputSize = "out_hidden_size"
        case projectorSize = "projector_hidden_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        text = try c.decode(Text.self, forKey: .text)
        vision = try c.decodeIfPresent(Vision.self, forKey: .vision)
        audio = try c.decodeIfPresent(Audio.self, forKey: .audio)
        imageToken = try c.decodeIfPresent(Int.self, forKey: .imageToken) ?? 200092
        videoToken = try c.decodeIfPresent(Int.self, forKey: .videoToken) ?? 200091
        audioToken = try c.decodeIfPresent(Int.self, forKey: .audioToken) ?? 200038
        let merged =
            vision.map {
                $0.geometry.hiddenSize * $0.geometry.mergeSize * $0.geometry.mergeSize
            } ?? 0
        visionOutputSize = try c.decodeIfPresent(Int.self, forKey: .visionOutputSize) ?? merged
        projectorSize = try c.decodeIfPresent(Int.self, forKey: .projectorSize) ?? merged
        guard vision == nil || (visionOutputSize == merged && projectorSize > 0),
            audio == nil || audio?.outputSize == text.hiddenSize
        else {
            throw ModelExecutionError.invalidOptions(
                "Muse Spark encoder dimensions must match the model")
        }
    }

    static func matches(_ modelType: String) -> Bool {
        modelType == "muse_spark" || modelType == "muse_spark_open"
    }

    /// Some conversions keep tower config and processor metadata after dropping the weights.
    static func resolvingTowers(_ data: Data, weightNames: Set<String>) throws -> Data {
        guard !weightNames.isEmpty else { return data }
        var config = try JSONDecoder.json5().decode([String: JSONValue].self, from: data)
        for (field, tower) in [
            ("vision_config", "vision_tower."), ("audio_config", "audio_tower."),
        ] {
            if !weightNames.contains(where: {
                $0.hasPrefix(tower) || $0.hasPrefix("model." + tower)
            }) {
                config.removeValue(forKey: field)
            }
        }
        return try JSONEncoder().encode(config)
    }
}

private func validPositive(_ value: Float) -> Bool { value.isFinite && value > 0 }
