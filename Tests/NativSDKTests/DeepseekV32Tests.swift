import Foundation
import MLX
import MLXNN
import Testing

@testable import NativSDK

private func deepseekV32Config(_ json: String? = nil) throws -> DeepseekV32Configuration {
    let data =
        try json.map { Data($0.utf8) }
        ?? ModelTestConfigurations.configuration("deepseek_v32", removing: ["rope_theta"])
    return try JSONDecoder.json5().decode(DeepseekV32Configuration.self, from: data)
}

@Suite("DeepSeek V3.2")
struct DeepseekV32Tests {

    /// transformers 5.x nests rope, and theta comes from the nested block.
    @Test("Configuration reads nested rope parameters")
    func configuration() throws {
        let config = try deepseekV32Config()
        #expect(config.ropeTheta == 50000)
        #expect(config.ropeScaling?["rope_theta"]?.asFloat() == 50000)
        #expect(config.indexTopK == 4)

        // The first layers are dense; the rest route.
        #expect(config.isMoELayer(0) == false)
        #expect(config.isMoELayer(1))

        // A flat rope_theta still wins when there is no nested block.
        let flat = try deepseekV32Config(#"{"rope_theta": 12345.0}"#)
        #expect(flat.ropeTheta == 12345)
    }

    /// The indexer returns nothing while the cache still fits inside `index_topk`, since
    /// every position is attended anyway; past that it selects.
    @Test("The indexer engages only once the cache outgrows the budget")
    func indexerEngages() throws {
        let config = try deepseekV32Config()
        let indexer = DeepseekV32Indexer(config)
        let cache = KVCacheSimple()

        // Four positions is exactly the budget, so nothing is selected.
        let short = indexer(
            MLXArray.zeros([1, 4, 32]), queryRank: MLXArray.zeros([1, 4, 24]),
            mask: nil, cache: cache)
        #expect(short == nil)

        // Two more pushes the cache past it.
        let selected = indexer(
            MLXArray.zeros([1, 2, 32]), queryRank: MLXArray.zeros([1, 2, 24]),
            mask: nil, cache: cache)
        let indices = try #require(selected)
        eval(indices)
        #expect(indices.dim(-1) == 4)
    }

    /// `noaux_tc` routing: the correction bias steers selection but not the returned
    /// scores, and group pruning zeroes whole groups before the top-k.
    @Test("Expert selection uses the bias for choosing but not for scoring")
    func expertSelection() {
        let gates = MLXArray(
            [Float(0), 1, 2, 3, 4, 5, 6, 7], [1, 8])
        let bias = MLXArray(Array(repeating: Float(0), count: 8))

        let (indices, scores) = deepseekV32GroupExpertSelect(
            gates: gates, correctionBias: bias, topK: 2, groups: 1, topkGroups: 1,
            routedScalingFactor: 1, normTopkProb: false)
        eval(indices, scores)
        // The two largest gates win.
        #expect(Set(indices.asArray(Int32.self)) == Set([Int32(6), 7]))
        // Scores are the sigmoid of the gates, without the bias folded in.
        let expected = Array(indices.asArray(Int32.self).map { 1 / (1 + exp(-Float($0))) })
        for (actual, want) in zip(scores.asArray(Float.self), expected) {
            #expect(abs(actual - want) < 1e-5)
        }

        // A bias that favours the low experts changes the choice, not the scores.
        let steered = MLXArray([Float(10), 10, 0, 0, 0, 0, 0, 0])
        let (steeredIndices, steeredScores) = deepseekV32GroupExpertSelect(
            gates: gates, correctionBias: steered, topK: 2, groups: 1, topkGroups: 1,
            routedScalingFactor: 1, normTopkProb: false)
        eval(steeredIndices, steeredScores)
        #expect(Set(steeredIndices.asArray(Int32.self)) == Set([Int32(0), 1]))
        #expect(steeredScores.asArray(Float.self).allSatisfy { $0 < 0.8 })

        // Normalizing makes the kept scores a distribution, then scaled.
        let (_, normalized) = deepseekV32GroupExpertSelect(
            gates: gates, correctionBias: bias, topK: 2, groups: 1, topkGroups: 1,
            routedScalingFactor: 2, normTopkProb: true)
        eval(normalized)
        #expect(abs(normalized.sum().item(Float.self) - 2) < 1e-5)
    }

    /// A published checkpoint fuses `kv_b_proj`; the absorbed path needs it split.
    @Test("Sanitize splits the fused KV projection and stacks experts")
    func sanitize() throws {
        let config = try deepseekV32Config()
        let model = DeepseekV32Model(config)
        let headDim = config.qkNopeHeadDim + config.vHeadDim

        var weights: [String: MLXArray] = [
            "model.layers.0.self_attn.kv_b_proj.weight": MLXArray.zeros([
                config.attentionHeads * headDim, config.kvLoraRank,
            ])
        ]
        for expert in 0 ..< 4 {
            for projection in ["gate_proj", "down_proj", "up_proj"] {
                weights["model.layers.1.mlp.experts.\(expert).\(projection).weight"] =
                    MLXArray.zeros([32, 32])
            }
        }
        // A layer past the stack belongs to the MTP drafter, not here.
        weights["model.layers.2.self_attn.kv_b_proj.weight"] = MLXArray.zeros([4, 4])

        let sanitized = model.sanitize(weights: weights)

        #expect(sanitized["model.layers.0.self_attn.kv_b_proj.weight"] == nil)
        // embed_q is transposed into the latent's basis; unembed_out is not.
        #expect(
            sanitized["model.layers.0.self_attn.embed_q.weight"]?.shape
                == [config.attentionHeads, config.kvLoraRank, config.qkNopeHeadDim])
        #expect(
            sanitized["model.layers.0.self_attn.unembed_out.weight"]?.shape
                == [config.attentionHeads, config.vHeadDim, config.kvLoraRank])
        // Experts are stacked into the switched MLP.
        #expect(
            sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape
                == [4, 32, 32])
        #expect(!sanitized.keys.contains { $0.contains("mlp.experts.") })
        #expect(!sanitized.keys.contains { $0.hasPrefix("model.layers.2.") })
    }

    /// The parameter tree is what a checkpoint has to fill, so its names are the contract.
    @Test("Parameter names follow the published layout")
    func parameterNames() throws {
        let model = DeepseekV32Model(try deepseekV32Config())
        let names = Set(model.parameters().flattened().map(\.0))

        for name in [
            "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
            "model.layers.0.self_attn.q_a_proj.weight",
            "model.layers.0.self_attn.q_a_layernorm.weight",
            "model.layers.0.self_attn.q_b_proj.weight",
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.0.self_attn.kv_a_layernorm.weight",
            "model.layers.0.self_attn.embed_q.weight",
            "model.layers.0.self_attn.unembed_out.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.indexer.wq_b.weight",
            "model.layers.0.self_attn.indexer.wk.weight",
            "model.layers.0.self_attn.indexer.k_norm.weight",
            "model.layers.0.self_attn.indexer.weights_proj.weight",
            "model.layers.0.input_layernorm.weight",
        ] {
            #expect(names.contains(name), "missing \(name)")
        }
        // Layer 0 is dense, layer 1 routes.
        #expect(names.contains("model.layers.0.mlp.gate_proj.weight"))
        #expect(names.contains("model.layers.1.mlp.switch_mlp.gate_proj.weight"))
        #expect(names.contains("model.layers.1.mlp.gate.e_score_correction_bias"))
        #expect(names.contains("model.layers.1.mlp.shared_experts.gate_proj.weight"))
    }
}
