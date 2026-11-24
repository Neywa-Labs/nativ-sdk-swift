import Foundation
import MLX

enum Ernie55Checkpoint {
    static let sourceQuantization = BaseConfiguration.Quantization(
        groupSize: 32, bits: 8, mode: .mxfp8)

    private static let echoNames = [
        "enorm.weight": "enorm.weight",
        "hnorm.weight": "hnorm.weight",
        "e_proj.weight": "e_proj.weight",
        "h_proj.weight": "h_proj.weight",
        "e_proj.scales": "e_proj.scales",
        "h_proj.scales": "h_proj.scales",
        "e_proj.biases": "e_proj.biases",
        "h_proj.biases": "h_proj.biases",
        "hc_head_fn": "hc_head.fn",
        "hc_head_base": "hc_head.base",
        "hc_head_scale": "hc_head.scale",
        "shared_head.norm.weight": "norm.weight",
        "block_sparse_moe.e_score_correction_bias": "e_score_correction_bias",
    ]

    private static let remaps = [
        (".input_layernorm.", ".attn_norm."),
        (".post_attention_layernorm.", ".ffn_norm."),
        (".block_sparse_moe.", ".ffn."),
        (".mlp.", ".ffn."),
        (".self_attn.", ".attn."),
        (".core_attention.", "."),
        (".linear_q_down_proj.", ".wq_a."),
        (".linear_q_up_proj.", ".wq_b."),
        (".linear_kv_proj.", ".wkv."),
        (".q_layernorm.", ".q_norm."),
        (".kv_layernorm.", ".kv_norm."),
        (".compressor.linear_wkv.", ".compressor.wkv."),
        (".compressor.linear_wgate.", ".compressor.wgate."),
        (".attn.gate_proj.", ".attn.o_gate."),
        (".attn.o_proj.", ".attn.wo_b."),
        (".attn.vha_postmix_U", ".attn.vha.U"),
        (".attn.vha_postmix_V", ".attn.vha.V"),
        (".attn.softmax_offset", ".attn.attn_sink"),
        (".shared_experts.w1.", ".shared_experts.gate_proj."),
        (".shared_experts.w2.", ".shared_experts.down_proj."),
        (".shared_experts.w3.", ".shared_experts.up_proj."),
    ]

    static func containsEcho(_ names: Set<String>, layers: Int) -> Bool {
        for (key, target) in echoNames {
            let native = "echo." + target
            let raw = "model.layers.\(layers)." + key
            if names.contains(native) || names.contains(raw)
                || names.contains("language_model." + native)
                || names.contains("language_model." + raw)
            {
                return true
            }
        }
        return false
    }

    /// The original checkpoint has E4M3 matrices with one floating scale per 128×128 block.
    /// Install large projections in MLX's per-row MXFP8 layout. Keep matrices that are
    /// rearranged for MLA or grouped output in BF16 until their layout is normalized.
    static func prepareSource(
        _ weights: inout [String: MLXArray], config: Ernie55Configuration
    ) throws -> Bool {
        let suffix = "_scale_inv"
        let keys = weights.keys.filter { $0.hasSuffix(suffix) }
        guard !keys.isEmpty else { return false }
        for scaleKey in keys {
            let key = String(scaleKey.dropLast(suffix.count))
            let normalized = withoutWrapper(key)
            if let layer = layerIndex(normalized), layer >= config.layers {
                let prefix = "model.layers.\(config.layers)."
                if layer != config.layers
                    || echoNames[String(normalized.dropFirst(prefix.count))] == nil
                {
                    weights.removeValue(forKey: key)
                    weights.removeValue(forKey: scaleKey)
                    continue
                }
            }
            guard let raw = weights[key], let scale = weights[scaleKey],
                raw.dtype == .uint8, raw.ndim == 2,
                scale.dtype == .float32 || scale.dtype == .bfloat16 || scale.dtype == .float16,
                scale.shape == [(raw.dim(0) + 127) / 128, (raw.dim(1) + 127) / 128]
            else {
                throw ModelExecutionError.invalidOptions("Invalid ERNIE 5.5 FP8 tensor \(key)")
            }
            weights.removeValue(forKey: scaleKey)
            if key.hasSuffix(".weight"), raw.dim(1).isMultiple(of: 32),
                !key.hasSuffix(".kv_b_proj.weight"), normalized != "lm_head.weight"
            {
                let projection = key.dropLast("weight".count)
                if let exponents = blockExponents(scale) {
                    // Baidu stores E8M0 powers of two in floating-point tensors.
                    // Expanding their exponents preserves the original FP8 bytes.
                    let columns = repeated(exponents, count: 4, axis: 1)
                    weights[key] = raw.view(dtype: .uint32)
                    weights[projection + "scales"] =
                        repeated(columns, count: 128, axis: 0)[
                            ..<raw.dim(0), ..<(raw.dim(1) / 32)]
                } else {
                    let dense = MiMoV2.dequantizeFP8Block(raw, scale: scale)
                    let packed = quantized(dense, groupSize: 32, bits: 8, mode: .mxfp8)
                    weights[key] = packed.wq
                    weights[projection + "scales"] = packed.scales
                }
            } else {
                weights[key] = MiMoV2.dequantizeFP8Block(raw, scale: scale)
            }
        }
        return true
    }

    private static func blockExponents(_ scale: MLXArray) -> MLXArray? {
        var exponents: [UInt8] = []
        exponents.reserveCapacity(scale.size)
        for value in scale.asType(.float32, stream: .cpu).asArray(Float.self) {
            let bits = value.bitPattern
            let exponent = (bits >> 23) & 0xff
            guard (bits & 0x807f_ffff) == 0, exponent > 0, exponent < 255 else { return nil }
            exponents.append(UInt8(exponent))
        }
        return MLXArray(exponents, scale.shape)
    }

    static func sanitize(_ weights: [String: MLXArray], config: Ernie55Configuration)
        -> [String: MLXArray]
    {
        var result: [String: MLXArray] = [:]
        for (name, tensor) in weights {
            if let layer = layerIndex(name), layer >= config.layers {
                if layer == config.layers {
                    let prefix = "model.layers.\(layer)."
                    let key = String(name.dropFirst(prefix.count))
                    if let target = echoNames[key] {
                        result["echo." + target] = key == "hc_head_fn" ? tensor.T : tensor
                    }
                }
                continue
            }

            if name.hasPrefix("model.lm_head.") {
                result["multimax." + name.dropFirst("model.lm_head.".count)] = tensor
                continue
            }
            if name.hasPrefix("model.mhc_contract.hc_head_") {
                let tail = name.dropFirst("model.mhc_contract.hc_head_".count)
                result["model.hc_head." + tail] = tail == "fn" ? tensor.T : tensor
                continue
            }

            var key = name
            var value = tensor
            for (source, target) in [("self_attention", "attn"), ("mlp", "ffn")] {
                let prefix = ".\(source)_hyper_connection."
                guard key.contains(prefix) else { continue }
                key = key.replacingOccurrences(of: prefix, with: ".\(target)_hc.")
                if key.hasSuffix("mapping_proj.weight") {
                    key = key.replacingOccurrences(of: "mapping_proj.weight", with: "fn")
                    value = value.T
                }
                key = key.replacingOccurrences(of: ".\(target)_hc.bias", with: ".\(target)_hc.base")
            }
            for (source, target) in remaps where key.contains(source) {
                key = key.replacingOccurrences(of: source, with: target)
            }
            if key.hasSuffix(".attn.linear_o_group_proj") {
                key = key.replacingOccurrences(of: ".linear_o_group_proj", with: ".wo_a.weight")
                value = value.reshaped(config.outputGroups, config.outputRank, -1)
            }
            result[key] = value
        }

        for layer in 0 ..< config.layers {
            let prefix = "model.layers.\(layer)."
            for kind in ["attn_hc", "ffn_hc"] {
                let path = prefix + kind
                let names = ["pre", "post", "res"].map { path + ".alpha_" + $0 }
                let values = names.compactMap { result[$0]?.reshaped(-1) }
                if values.count == names.count {
                    for name in names { result.removeValue(forKey: name) }
                    result[path + ".scale"] = concatenated(values)
                }
            }
            stackExperts(&result, prefix: prefix + "ffn.", count: config.experts)
            if config.compression[layer] == -2 {
                splitMLA(&result, prefix: prefix + "attn.", config: config)
            }
        }
        return result
    }

    private static func stackExperts(
        _ weights: inout [String: MLXArray], prefix: String, count: Int
    ) {
        for (src, dst) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
            for kind in ["weight", "scales", "biases"] {
                let suffix = ".\(src).\(kind)"
                guard weights[prefix + "experts.0" + suffix] != nil else { continue }
                let names = (0 ..< count).map { prefix + "experts.\($0)" + suffix }
                let values = names.compactMap { weights[$0] }
                if values.count == names.count {
                    for name in names { weights.removeValue(forKey: name) }
                    weights[prefix + "switch_mlp.\(dst).\(kind)"] = stacked(values)
                }
            }
        }
    }

    private static func splitMLA(
        _ weights: inout [String: MLXArray], prefix: String, config: Ernie55Configuration
    ) {
        let projection = prefix + "kv_b_proj."
        guard var fused = weights.removeValue(forKey: projection + "weight") else { return }
        if let scales = weights.removeValue(forKey: projection + "scales") {
            let group = config.mlaLatent / scales.dim(-1)
            let bits = fused.dim(-1) * 32 / config.mlaLatent
            let mode: QuantizationMode =
                scales.dtype != .uint8
                ? .affine : (bits == 8 ? .mxfp8 : (group == 16 ? .nvfp4 : .mxfp4))
            fused = dequantized(
                fused, scales: scales, biases: weights.removeValue(forKey: projection + "biases"),
                groupSize: group, bits: bits, mode: mode, dtype: .bfloat16)
        }
        let shaped = fused.reshaped(
            config.mlaHeads, config.mlaNope + config.mlaValue, config.mlaLatent)
        weights[prefix + "embed_q.weight"] = shaped[0..., ..<config.mlaNope, 0...].swappedAxes(
            -1, -2)
        weights[prefix + "unembed_out.weight"] = shaped[0..., config.mlaNope..., 0...]
    }

    private static func layerIndex(_ key: String) -> Int? {
        guard key.hasPrefix("model.layers.") else { return nil }
        return Int(key.dropFirst("model.layers.".count).prefix { $0 != "." })
    }

    private static func withoutWrapper(_ path: String) -> String {
        let prefix = "language_model."
        return path.hasPrefix(prefix) ? String(path.dropFirst(prefix.count)) : path
    }
}
