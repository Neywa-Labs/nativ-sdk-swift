import Foundation
import MLX

enum Inkling2Checkpoint {
    private static let language = "language_model.model."
    private static let sourceLanguage = "model.llm."
    private static let sourceVision = "model.visual."

    private struct QuantizationMetadata: Decodable {
        struct Model: Decodable {
            struct ParameterGroup: Decodable {
                let weightTransform: String
                let tensors: [String]

                enum CodingKeys: String, CodingKey {
                    case weightTransform = "weight_transform"
                    case tensors
                }
            }

            let parameterGroups: [ParameterGroup]

            enum CodingKeys: String, CodingKey {
                case parameterGroups = "parameter_groups"
            }
        }

        let quantMethod: String
        let model: Model

        enum CodingKeys: String, CodingKey {
            case quantMethod = "quant_method"
            case model
        }
    }

    static func quantization(from directory: URL) throws -> BaseConfiguration.PerLayerQuantization?
    {
        let file = directory.appendingPathComponent("hf_quant_config.json")
        guard FileManager.default.fileExists(atPath: file.path) else { return nil }
        let metadata = try JSONDecoder().decode(
            QuantizationMetadata.self, from: Data(contentsOf: file))
        guard metadata.quantMethod == "inkling2" else {
            throw ModelExecutionError.invalidOptions(
                "Unsupported Inkling 2 checkpoint quantization.")
        }
        var options: [String: BaseConfiguration.QuantizationOption] = [:]
        for group in metadata.model.parameterGroups {
            let quantization: BaseConfiguration.Quantization
            switch group.weightTransform {
            case "cast": continue
            case "nvfp4_nearest": quantization = .init(groupSize: 16, bits: 4, mode: .nvfp4)
            case "mxfp8_nearest": quantization = .init(groupSize: 32, bits: 8, mode: .mxfp8)
            default:
                throw ModelExecutionError.invalidOptions(
                    "Unsupported Inkling 2 weight transform \(group.weightTransform).")
            }
            for name in group.tensors {
                let parts = name.split(separator: ".")
                guard parts.count == 7, parts[0] == "model", parts[1] == "llm",
                    parts[2] == "layers", let layer = Int(parts[3]), layer >= 0,
                    parts[4] == "mlp", parts[5] == "experts",
                    parts[6] == "w13_weight" || parts[6] == "w2_weight"
                else {
                    throw ModelExecutionError.invalidOptions(
                        "Invalid Inkling 2 expert tensor \(name).")
                }
                let projections =
                    parts[6] == "w13_weight" ? ["gate_proj", "up_proj"] : ["down_proj"]
                for projection in projections {
                    options[language + "layers.\(layer).mlp.switch_mlp." + projection] =
                        .quantize(quantization)
                }
            }
        }
        return .init(perLayerQuantization: options)
    }

    private static let attentionNames = [
        "wq_du": "q_proj", "wk_dv": "k_proj", "wv_dv": "v_proj", "wr_du": "r_proj",
        "wo_ud": "o_proj", "gate_attn": "gate_proj", "attn_sink_proj": "sink_proj",
    ]

    private static let layerNames = [
        "attn_norm": "input_layernorm", "mlp_norm": "post_attention_layernorm",
        "q_input_norm": "q_input_layernorm", "k_input_norm": "k_input_layernorm",
        "v_input_norm": "v_input_layernorm",
    ]

    private static let attentionParameters = [
        "q_l2_clip.scale": "q_scale", "rel_l2_clip.scale": "rel_scale",
        "rel_logits_proj.proj": "rel_proj",
        "k_sconv.static_w": "k_sconv.static_weight", "v_sconv.static_w": "v_sconv.static_weight",
        "k_sconv.weight_proj": "k_sconv.weight_proj.weight",
        "v_sconv.weight_proj": "v_sconv.weight_proj.weight",
        "k_sconv.weight": "k_sconv.conv.weight", "v_sconv.weight": "v_sconv.conv.weight",
    ]

    private static let layerParameters = [
        "mlp.gate.weight": "mlp.gate_weight", "mlp.gate.bias": "mlp.e_score_correction_bias",
        "attn_sconv.weight": "attn_sconv.conv.weight", "mlp_sconv.weight": "mlp_sconv.conv.weight",
    ]

    private static let visionNames = [
        ("ln_pre.", "pre_layernorm."), (".ln_1.", ".layernorm_before."),
        (".ln_2.", ".layernorm_after."), (".ls_1.gamma", ".lambda_1"),
        (".ls_2.gamma", ".lambda_2"), (".mlp.c_fc.", ".mlp.fc1."), (".mlp.c_proj.", ".mlp.fc2."),
    ]

    /// Map the published Thinking Machines weights to the same module names as mlx-vlm.
    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (name, weight) in weights {
            if name.hasPrefix("model.mtp.") || name.hasPrefix("mtp.") { continue }
            let parts = name.split(separator: ".")
            if name.hasPrefix(sourceLanguage + "layers."), parts.count >= 5,
                let index = Int(parts[3])
            {
                let prefix = language + "layers.\(index)."
                let leaf = parts.dropFirst(4).joined(separator: ".")
                if leaf.hasPrefix("mlp.experts.") {
                    // Handle fused expert projections once each from their weight entry.
                    if leaf == "mlp.experts.w13_weight" || leaf == "mlp.experts.w2_weight" {
                        splitExperts(
                            source: name, target: prefix + "mlp.switch_mlp.",
                            gateUp: leaf.hasSuffix("w13_weight"), weights: weights, result: &result)
                    }
                    continue
                }
                let target: String
                if parts[4] == "attn", parts.count >= 6 {
                    let parameter = parts.dropFirst(5).joined(separator: ".")
                    let projection =
                        (attentionNames[String(parts[5])] ?? String(parts[5]))
                        + parts.dropFirst(6).map { "." + $0 }.joined()
                    target = "self_attn." + (attentionParameters[parameter] ?? projection)
                } else {
                    let module = layerNames[String(parts[4])] ?? String(parts[4])
                    target =
                        layerParameters[leaf] ?? module
                        + parts.dropFirst(5).map { "." + $0 }.joined()
                }
                result[prefix + target] =
                    target.hasSuffix(".conv.weight") && weight.ndim == 3 && weight.dim(-1) != 1
                    ? weight.transposed(0, 2, 1) : weight
            } else if name.hasPrefix(sourceLanguage + "attn_res.") {
                if parts.last == "output_query" {
                    result[language + "final_input.query"] = weight
                } else if parts.count == 5, let index = Int(parts[4]) {
                    let attention = parts[3] == "attn_queries"
                    let layer = index + (attention ? 1 : 0)
                    let input = attention ? "attention_input" : "mlp_input"
                    result[language + "layers.\(layer).\(input).query"] = weight
                } else {
                    result[name] = weight
                }
            } else if name.hasPrefix(sourceLanguage) {
                let leaf = String(name.dropFirst(sourceLanguage.count))
                switch leaf {
                case "unembed.weight": result["language_model.lm_head.weight"] = weight
                case "embed.weight": result[language + "embed_tokens.weight"] = weight
                default: result[language + leaf] = weight
                }
            } else if name.hasPrefix(sourceVision) {
                var leaf = String(name.dropFirst(sourceVision.count))
                if leaf.hasPrefix("backbone.") { leaf = String(leaf.dropFirst("backbone.".count)) }
                for (from, to) in visionNames {
                    leaf = leaf.replacingOccurrences(of: from, with: to)
                }
                result["vision_tower." + leaf] = weight
            } else {
                result[name] = weight
            }
        }
        return result
    }

    private static func splitExperts(
        source: String, target: String, gateUp: Bool,
        weights: [String: MLXArray], result: inout [String: MLXArray]
    ) {
        guard let weight = weights[source], weight.ndim == 3,
            !gateUp || weight.dim(1).isMultiple(of: 2)
        else { return }
        let scales = weights[source + "_scale"]
        let packed = scales != nil && weight.dtype == .uint8 ? weight.view(dtype: .uint32) : weight
        func store(_ value: MLXArray, as suffix: String) {
            if gateUp {
                // The reference stores alternating gate and up rows.
                let paired = value.reshaped(value.dim(0), value.dim(1) / 2, 2, value.dim(2))
                result[target + "gate_proj." + suffix] = contiguous(paired[.ellipsis, 0, 0...])
                result[target + "up_proj." + suffix] = contiguous(paired[.ellipsis, 1, 0...])
            } else {
                result[target + "down_proj." + suffix] = value
            }
        }
        store(packed, as: "weight")
        if let scales, scales.ndim == 3, scales.dim(1) == weight.dim(1) {
            store(scales, as: "scales")
        }
        if let amax = weights[source + "_amax"],
            let denominator = weights[source + "_outer_scale_denominator"]
        {
            let scale = (amax.asType(.float32) / denominator.asType(.float32)).reshaped(-1)
            for projection in gateUp ? ["gate_proj", "up_proj"] : ["down_proj"] {
                result[target + projection + ".weight_scale_2"] = scale
            }
        }
    }
}
