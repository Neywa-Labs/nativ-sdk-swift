import Foundation
import MLX
import XCTest

@testable import NativSDK

/// Small model architectures for cache, checkpoint and processor behavior tests.
/// Weights are initialized locally; no recorded numerical outputs are needed.
enum ModelTestConfigurations {
    static let textModels = [
        "cohere2", "spark2_5", "mellum", "exaone_moe", "minicpm3", "youtu_llm",
        "hunyuan", "afm7", "nemotron-nas", "iquestloopcoder", "ministral3", "mamba",
        "internlm3", "phi3small",
    ]
    static let additionalTextModels = [
        "falcon_mamba", "youtu_llm_direct", "hunyuan_dense", "hunyuan_default_expert_width",
    ]
    static let extendedModels = [
        "bailing_moe_linear", "deepseek_v32", "glm_moe_dsa", "hy_v4", "inkling", "kimi_linear",
        "limite", "longcat_flash", "longcat_flash_ngram", "longcat_flash_sparse", "plamo2",
        "recurrent_gemma", "rwkv7", "step3p5", "z1t",
    ]

    static func configuration(
        _ name: String, overrides: [String: Any] = [:], removing: [String] = []
    ) throws -> Data {
        let base: String
        switch name {
        case "falcon_mamba": base = "mamba"
        case "youtu_llm_direct": base = "youtu_llm"
        case "hunyuan_dense", "hunyuan_default_expert_width": base = "hunyuan"
        default: base = name
        }
        let url = try XCTUnwrap(Bundle.module.resourceURL)
            .appendingPathComponent("Fixtures/model-configurations.json")
        let catalog = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: [String: Any]])
        func expanded(_ values: [String: Any]) throws -> [String: Any] {
            var result: [String: Any] = [:]
            if let baseName = values["_base"] as? String {
                result = try expanded(XCTUnwrap(catalog[baseName]))
            }
            for (key, value) in values where key != "_base" {
                result[key] = try (value as? [String: Any]).map(expanded) ?? value
            }
            return result
        }
        var values = try expanded(XCTUnwrap(catalog[base], "Unknown model fixture: \(name)"))
        switch name {
        case "falcon_mamba":
            values["model_type"] = name
            values["use_bcdt_rms"] = false
        case "youtu_llm_direct": values["q_lora_rank"] = NSNull()
        case "hunyuan_dense": values["num_experts"] = 1
        case "hunyuan_default_expert_width": values.removeValue(forKey: "moe_intermediate_size")
        default: break
        }
        values.merge(overrides) { _, new in new }
        for key in removing { values.removeValue(forKey: key) }
        return try JSONSerialization.data(withJSONObject: values)
    }

    static func model(_ name: String) throws -> any LanguageModel {
        let data = try configuration(name)
        let config = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let model = try ModelRegistry.default.createModel(
            configuration: data, modelType: XCTUnwrap(config["model_type"] as? String))
        model.train(false)
        eval(model)
        return model
    }
    static func textConfiguration(_ modelType: String, tiedEmbeddings: Bool = false) throws -> Data
    {
        var values =
            try XCTUnwrap(
                JSONSerialization.jsonObject(with: configuration("_text")) as? [String: Any])
        values["model_type"] = modelType
        values["tie_word_embeddings"] = tiedEmbeddings
        if ["granitemoehybrid", "lfm2", "lfm2_moe"].contains(modelType) {
            values["layer_types"] =
                modelType == "granitemoehybrid"
                ? ["mamba", "attention"] : ["conv", "full_attention"]
        }
        if ["qwen3_5", "qwen3_5_text"].contains(modelType) { values["num_experts"] = 0 }
        values["use_qk_norm"] = true
        values["use_qkv_bias"] = false
        values["use_expert_bias"] = false
        values["moe_router_enable_expert_bias"] = false
        values["conv_bias"] = false
        values["attn_layer_offset"] = 1
        values["attn_layer_period"] = 2
        values["expert_layer_offset"] = 0
        values["expert_layer_period"] = 1
        if modelType == "exaone4" { values["sliding_window_pattern"] = "LG" }
        if ["glm4_moe", "glm4_moe_lite", "mimo_v2_flash"].contains(modelType) {
            values["topk_method"] = "noaux_tc"
            values["scoring_func"] = "sigmoid"
        }
        if modelType == "mimo_v2_flash" { values["moe_layer_freq"] = [1, 1] }
        if modelType == "mamba2" { values["head_dim"] = 16 }
        if modelType == "phimoe" {
            values["rope_scaling"] = [
                "short_factor": Array(repeating: 1.0, count: 8),
                "long_factor": Array(repeating: 1.0, count: 8),
            ]
        }
        return try JSONSerialization.data(withJSONObject: values)
    }

}
