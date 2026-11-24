import Foundation
import MLX
import MLXNN
import Testing

@testable import NativSDK

private func glmMoeDsaConfig(_ json: String? = nil) throws -> GlmMoeDsaConfiguration {
    let data =
        try json.map { Data($0.utf8) }
        ?? ModelTestConfigurations.configuration(
            "glm_moe_dsa",
            removing: [
                "topk_method", "scoring_func", "norm_topk_prob", "moe_layer_freq", "attention_bias",
            ])
    return try JSONDecoder.json5().decode(GlmMoeDsaConfiguration.self, from: data)
}

@Suite("GLM MoE DSA")
struct GlmMoeDsaTests {

    /// With no explicit schedule, every `freq`th layer from the offset runs its own
    /// indexer and the rest reuse the last selection.
    @Test("The indexer schedule is derived from frequency and offset")
    func derivedSchedule() throws {
        let config = try glmMoeDsaConfig()
        // (i - offset + 1) % freq == 0, clamped at zero: layers 1 and 3 here.
        #expect(config.indexerTypes == ["full", "full", "shared", "full"])
        #expect(config.usesFullIndexer(layerIndex: 0))
        #expect(config.usesFullIndexer(layerIndex: 2) == false)
        #expect(config.nextnPredictLayers == 1)
        // The DeepSeek half decodes as its own config.
        #expect(config.deepseek.ropeTheta == 50000)
        #expect(config.deepseek.indexTopK == 4)
    }

    /// A string pattern spells the schedule out directly, and an explicit list wins over
    /// both.
    @Test("An explicit pattern or list overrides the derivation")
    func explicitSchedule() throws {
        let pattern = try glmMoeDsaConfig(
            #"{"num_hidden_layers": 4, "index_topk_pattern": "FSSF"}"#)
        #expect(pattern.indexerTypes == ["full", "shared", "shared", "full"])

        let list = try glmMoeDsaConfig(
            #"{"num_hidden_layers": 2, "indexer_types": ["shared", "full"]}"#)
        #expect(list.indexerTypes == ["shared", "full"])
        #expect(list.usesFullIndexer(layerIndex: 0) == false)
    }

    /// A prompt pass and a decode step both run, which exercises the shared-indexer layer:
    /// it has no second cache to reach for and takes the previous layer's selection.
    @Test("A stack mixing full and shared indexers produces logits")
    func forward() throws {
        let config = try glmMoeDsaConfig()
        let model = GlmMoeDsaModel(config)
        let cache = try model.newCache(configuration: nil)

        let prompt = MLXArray((0 ..< 6).map { Int32($0) }).reshaped(1, 6)
        let logits = model(prompt, cache: cache)
        eval(logits)
        #expect(logits.shape == [1, 6, 64])

        let step = model(MLXArray([Int32(2)]).reshaped(1, 1), cache: cache)
        eval(step)
        #expect(step.shape == [1, 1, 64])

        // Every layer advanced, including the one with a single cache.
        for entry in cache {
            let list = try #require(entry as? CacheList)
            #expect(list[0].offset == 7)
        }
    }

}
