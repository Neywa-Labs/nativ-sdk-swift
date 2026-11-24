import Foundation
import MLX

enum MistralLarge4Checkpoint {
    private static let projections = ["w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"]

    static func isDraftWeight(_ name: String) -> Bool {
        name.hasPrefix("mtp") || name.contains(".mtp.")
    }

    static func normalize(
        _ raw: [String: MLXArray], config: MistralLarge4Configuration, vision: MistralLarge4?
    ) throws -> [String: MLXArray] {
        var language = [String: MLXArray]()
        var image = [String: MLXArray]()
        var projector = [String: MLXArray]()
        let projectorAliases = [
            "pre_mm_projector_norm.weight": "norm.weight",
            "patch_merger.merging_layer.weight": "patch_merger.merging_layer.weight",
            "vision_language_adapter.w_in.weight": "linear_1.weight",
            "vision_language_adapter.w_out.weight": "linear_2.weight",
            "vision_language_adapter.w_in.bias": "linear_1.bias",
            "vision_language_adapter.w_out.bias": "linear_2.bias",
        ]
        let visionRenames = [
            (".attention.wq.", ".attention.q_proj."), (".attention.wk.", ".attention.k_proj."),
            (".attention.wv.", ".attention.v_proj."), (".attention.wo.", ".attention.o_proj."),
            (".feed_forward.w1.", ".feed_forward.gate_proj."),
            (".feed_forward.w2.", ".feed_forward.down_proj."),
            (".feed_forward.w3.", ".feed_forward.up_proj."),
        ]
        for (name, value) in raw {
            let key =
                name.hasPrefix("model.")
                    && [
                        "model.language_model.", "model.vision_tower.",
                        "model.multi_modal_projector.",
                    ].contains(where: name.hasPrefix)
                ? String(name.dropFirst("model.".count)) : name
            if isDraftWeight(key)
                || [".input_scale", ".k_scale", ".v_scale"].contains(where: key.hasSuffix)
            {
                continue
            }
            if let mapped = projectorAliases[key] {
                projector[mapped] = value
            } else if key.hasPrefix("multi_modal_projector.") {
                projector[String(key.dropFirst("multi_modal_projector.".count))] = value
            } else if key.hasPrefix("vision_encoder.") {
                let path = "vision_model." + String(key.dropFirst("vision_encoder.".count))
                let renamed = visionRenames.reduce(path) {
                    $0.replacingOccurrences(of: $1.0, with: $1.1)
                }
                image[renamed] = value
            } else if key.hasPrefix("vision_tower.") {
                image[String(key.dropFirst("vision_tower.".count))] = value
            } else {
                let path = languageKey(key)
                if config.text.tiedEmbeddings, path.hasPrefix("lm_head.") { continue }
                language[path] = value
            }
        }
        try normalizeStorage(&language, block: config.fp8Block)
        try normalizeLanguage(&language, config: config.text)
        guard let vision else { return language }

        var result = Dictionary(
            uniqueKeysWithValues: language.map { ("language_model." + $0.key, $0.value) })
        for (key, value) in vision.vision.sanitize(weights: image) {
            result["vision_tower." + key] = value
        }
        for (key, value) in projector { result["multi_modal_projector." + key] = value }
        return result
    }

    private static func languageKey(_ original: String) -> String {
        let key =
            original.hasPrefix("language_model.")
            ? String(original.dropFirst("language_model.".count)) : original
        switch key {
        case "tok_embeddings.weight": return "model.embed_tokens.weight"
        case "norm.weight": return "model.norm.weight"
        case "output.weight": return "lm_head.weight"
        case "lerope.alpha", "model.rope.alpha": return "model.rotary_emb.alpha"
        default: break
        }
        let bare = key.hasPrefix("model.") ? String(key.dropFirst("model.".count)) : key
        let parts = bare.split(separator: ".", maxSplits: 2).map(String.init)
        guard parts.count == 3, parts[0] == "layers" else { return key }
        var rest = parts[2]
        if rest.hasPrefix("attention.") {
            rest = "self_attn." + String(rest.dropFirst("attention.".count))
        }
        if rest.hasPrefix("feed_forward.") {
            rest = "mlp." + String(rest.dropFirst("feed_forward.".count))
        }
        if ["experts.", "shared_experts.", "gate."].contains(where: rest.hasPrefix) {
            rest = "mlp." + rest
        }
        let aliases = [
            ("attention_norm.", "input_layernorm."), ("ffn_norm.", "post_attention_layernorm."),
            ("self_attn.wq.", "self_attn.q_proj."), ("self_attn.wo.", "self_attn.o_proj."),
            ("self_attn.wkv_a_with_mqa.", "self_attn.kv_a_proj_with_mqa."),
            ("self_attn.kv_a_norm.", "self_attn.kv_a_layernorm."),
            ("self_attn.wkv_b.", "self_attn.kv_b_proj."),
        ]
        if let (source, destination) = aliases.first(where: { rest.hasPrefix($0.0) }) {
            rest = destination + String(rest.dropFirst(source.count))
        }
        rest = rest.replacingOccurrences(of: ".ssmax_scaling.b", with: ".softmax_scaling.bias")
            .replacingOccurrences(of: ".ssmax_scaling.s", with: ".softmax_scaling.scale")
        if rest.hasPrefix("mlp.") {
            for (native, canonical) in projections {
                rest = rest.replacingOccurrences(of: ".\(native).", with: ".\(canonical).")
            }
        }
        return "model.layers.\(parts[1]).\(rest)"
    }

    private static func normalizeStorage(_ weights: inout [String: MLXArray], block: [Int]) throws {
        for key in Array(weights.keys) {
            guard
                let suffix = [".qscale_weight", ".weight_scale_inv", ".weight_scale"]
                    .first(where: key.hasSuffix),
                let scale = weights.removeValue(forKey: key)
            else { continue }
            let path = String(key.dropLast(suffix.count))
            guard let weight = weights[path + ".weight"], weight.dtype == .uint8, weight.ndim == 2
            else {
                throw ModelExecutionError.invalidOptions(
                    "Invalid Mistral Large 4 quantized matrix \(path)")
            }

            if let global = weights[path + ".weight_scale_2"] {
                guard suffix == ".weight_scale", scale.dtype == .uint8,
                    weight.dim(1).isMultiple(of: 8), global.size == 1,
                    scale.shape == [weight.dim(0), weight.dim(1) / 8]
                else {
                    throw ModelExecutionError.invalidOptions(
                        "Invalid Mistral Large 4 NVFP4 scales for \(path)")
                }
                weights[path + ".weight"] = weight.view(dtype: .uint32)
                weights[path + ".scales"] = scale
                weights[path + ".weight_scale_2"] = global.asType(.float32).reshaped([])
                continue
            }

            let (rows, columns) = (weight.dim(0), weight.dim(1))
            let (blockRows, blockColumns) = (block[0], block[1])
            let groups = [(rows - 1) / blockRows + 1, (columns - 1) / blockColumns + 1]
            guard scale.dtype.isFloatingPoint,
                scale.shape == groups || scale.shape == [groups[0], 1, groups[1], 1]
            else {
                throw ModelExecutionError.invalidOptions(
                    "Invalid Mistral Large 4 FP8 block scales for \(path)")
            }
            let paddedRows = groups[0] * blockRows
            let paddedColumns = groups[1] * blockColumns
            let bytes = padded(
                weight,
                widths: [.init((0, paddedRows - rows)), .init((0, paddedColumns - columns))])
            let decoded = fromFP8(bytes, dtype: .float32)
                .reshaped(groups[0], blockRows, groups[1], blockColumns)
            let matrix = (decoded * scale.asType(.float32).reshaped(groups[0], 1, groups[1], 1))
                .reshaped(paddedRows, paddedColumns)[..<rows, ..<columns].asType(.bfloat16)
            eval(matrix)
            weights[path + ".weight"] = matrix
        }
    }

    private static func normalizeLanguage(
        _ weights: inout [String: MLXArray], config: MistralLarge4Configuration.Text
    ) throws {
        for layer in 0 ..< config.layers {
            let base = "model.layers.\(layer)"
            let attention = base + ".self_attn"
            let mlp = base + ".mlp"
            if let fused = weights.removeValue(forKey: attention + ".kv_b_proj.weight") {
                let width = config.contentSize + (config.rotary[layer] ? 0 : config.ropeSize)
                guard fused.dtype.isFloatingPoint,
                    fused.shape == [config.heads * (width + config.valueSize), config.latentSize]
                else {
                    throw ModelExecutionError.invalidOptions(
                        "Invalid Mistral Large 4 KV projection at layer \(layer)")
                }
                let splitKV = fused.reshaped(
                    config.heads, width + config.valueSize, config.latentSize)
                weights[attention + ".embed_q.weight"] = contiguous(
                    splitKV[0..., ..<width, 0...].swappedAxes(-1, -2))
                weights[attention + ".unembed_out.weight"] = contiguous(
                    splitKV[0..., width..., 0...])
            }
            splitProjection(
                &weights, prefix: attention + ".indexer", from: "wk_weights_proj",
                first: "wk", second: "weights_proj", splitAt: config.indexSize)
            if layer < config.denseLayers {
                splitProjection(
                    &weights, prefix: mlp, from: "gate_up_proj",
                    first: "gate_proj", second: "up_proj", splitAt: config.intermediateSize)
                continue
            }
            splitProjection(
                &weights, prefix: mlp + ".shared_experts", from: "gate_up_proj",
                first: "gate_proj", second: "up_proj",
                splitAt: config.expertSize * config.sharedExperts)
            let source = mlp + ".experts"
            let target = mlp + ".switch_mlp"
            if let packed = weights.removeValue(forKey: source + ".gate_up_proj") {
                let parts = split(packed, indices: [config.expertSize], axis: -2)
                weights[target + ".gate_proj.weight"] = parts[0]
                weights[target + ".up_proj.weight"] = parts[1]
            }
            if let down = weights.removeValue(forKey: source + ".down_proj") {
                weights[target + ".down_proj.weight"] = down
            }
            for projection in projections.values {
                for suffix in ["weight", "scales", "biases", "weight_scale_2"] {
                    let tail = ".\(projection).\(suffix)"
                    if let alreadyStacked = weights.removeValue(forKey: source + tail) {
                        weights[target + tail] = alreadyStacked
                    } else if weights[source + ".0" + tail] != nil {
                        var parts = [MLXArray]()
                        for expert in 0 ..< config.experts {
                            guard
                                let value = weights.removeValue(
                                    forKey: "\(source).\(expert)" + tail)
                            else {
                                throw ModelExecutionError.invalidOptions(
                                    "Missing Mistral Large 4 expert \(expert) in layer \(layer)")
                            }
                            parts.append(value)
                        }
                        weights[target + tail] = stacked(parts)
                    }
                }
            }
        }
    }

    private static func splitProjection(
        _ weights: inout [String: MLXArray], prefix: String, from: String,
        first: String, second: String, splitAt: Int
    ) {
        for suffix in ["weight", "scales", "biases"] {
            guard let joined = weights.removeValue(forKey: "\(prefix).\(from).\(suffix)") else {
                continue
            }
            let parts = split(joined, indices: [splitAt], axis: -2)
            weights["\(prefix).\(first).\(suffix)"] = parts[0]
            weights["\(prefix).\(second).\(suffix)"] = parts[1]
        }
    }
}
