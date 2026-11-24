import Foundation
import MLX
import MLXNN

/// GLM's MoE with DeepSeek sparse attention.
///
/// Everything but the indexer schedule is DeepSeek V3.2's, so the config carries that one
/// and adds the schedule on top.
public struct GlmMoeDsaConfiguration: Decodable, Sendable {
    public var deepseek: DeepseekV32Configuration
    /// Per layer, `"full"` when the layer runs its own indexer and `"shared"` when it
    /// reuses the selection made by the last layer that did.
    public var indexerTypes: [String]
    public var nextnPredictLayers: Int = 0
    public var indexShareForMTPIteration: Bool = false

    enum CodingKeys: String, CodingKey {
        case indexerTypes = "indexer_types"
        case indexTopKPattern = "index_topk_pattern"
        case indexTopKFrequency = "index_topk_freq"
        case indexSkipTopKOffset = "index_skip_topk_offset"
        case nextnPredictLayers = "num_nextn_predict_layers"
        case indexShareForMTPIteration = "index_share_for_mtp_iteration"
    }

    public init(from decoder: Decoder) throws {
        deepseek = try DeepseekV32Configuration(from: decoder)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        nextnPredictLayers =
            try c.decodeIfPresent(Int.self, forKey: .nextnPredictLayers) ?? nextnPredictLayers
        indexShareForMTPIteration =
            try c.decodeIfPresent(Bool.self, forKey: .indexShareForMTPIteration)
            ?? indexShareForMTPIteration

        let layers = deepseek.hiddenLayers
        if let declared = try c.decodeIfPresent([String].self, forKey: .indexerTypes) {
            indexerTypes = declared
        } else if let pattern = try? c.decode(String.self, forKey: .indexTopKPattern) {
            // A string pattern spells the schedule out: F for full, S for shared.
            indexerTypes = pattern.map { $0 == "F" ? "full" : "shared" }
        } else if let pattern = try? c.decode([String].self, forKey: .indexTopKPattern) {
            indexerTypes = pattern
        } else {
            // Otherwise every `freq`th layer from the offset runs its own indexer.
            let frequency = max(
                try c.decodeIfPresent(Int.self, forKey: .indexTopKFrequency) ?? 1, 1)
            let offset = try c.decodeIfPresent(Int.self, forKey: .indexSkipTopKOffset) ?? 2
            indexerTypes = (0 ..< layers).map { index in
                max(index - offset + 1, 0) % frequency == 0 ? "full" : "shared"
            }
        }
    }

    /// Whether a layer runs its own indexer rather than reusing the previous selection.
    public func usesFullIndexer(layerIndex: Int) -> Bool {
        guard layerIndex < indexerTypes.count else { return true }
        return indexerTypes[layerIndex] != "shared"
    }
}

public final class GlmMoeDsaModel: Module, LLMModel, KVCacheDimensionProvider {
    @ModuleInfo var model: DeepseekV32Inner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: GlmMoeDsaConfiguration
    public var vocabularySize: Int { config.deepseek.vocabularySize }
    public var kvHeads: [Int] {
        Array(repeating: 1, count: config.deepseek.hiddenLayers)
    }

    public init(_ config: GlmMoeDsaConfiguration) {
        self.config = config
        // Only the indexer schedule differs from DeepSeek V3.2, and it is a per-layer
        // decision, so the stack is built from that model's layers with the schedule
        // injected rather than restated here.
        _model.wrappedValue = DeepseekV32Inner(config.deepseek) { deepseek, index in
            DeepseekV32DecoderLayer(
                deepseek, layerIndex: index,
                makeAttention: { layerConfig, layerIndex in
                    DeepseekV32Attention(
                        layerConfig,
                        forceFullIndexer: config.usesFullIndexer(layerIndex: layerIndex))
                })
        }
        _lmHead.wrappedValue = Linear(
            config.deepseek.hiddenSize, config.deepseek.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    public var loraLayers: [Module] { model.layers }

    /// A shared-indexer layer keeps one cache; a layer with its own indexer keeps two.
    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        (0 ..< config.deepseek.hiddenLayers).map { index in
            config.usesFullIndexer(layerIndex: index)
                ? CacheList(
                    MaterializedKVCache(KVCacheSimple()), MaterializedKVCache(KVCacheSimple()))
                : CacheList(MaterializedKVCache(KVCacheSimple()))
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        DeepseekV32Model(config.deepseek).sanitize(weights: weights)
    }
}
