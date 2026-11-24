import Foundation
import MLX
import MLXNN
import Testing

@testable import NativSDK

private func inklingConfig(_ json: String) throws -> InklingTextConfiguration {
    try JSONDecoder.json5().decode(InklingTextConfiguration.self, from: Data(json.utf8))
}

/// Sliding and global layers have different geometry, not just a different mask, so the
/// config is read per layer.
private let tinyInkling = """
    {
      "model_type": "inkling",
      "hidden_size": 32, "num_hidden_layers": 2, "vocab_size": 64,
      "rms_norm_eps": 1e-6,
      "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
      "swa_num_attention_heads": 2, "swa_num_key_value_heads": 1, "swa_head_dim": 4,
      "sliding_window_size": 6, "d_rel": 2, "rel_extent": 5,
      "log_scaling_n_floor": 4, "log_scaling_alpha": 0.1,
      "sconv_kernel_size": 4,
      "layer_types": ["hybrid_sliding", "full_attention"]
    }
    """

private func layerCache() -> CacheList {
    CacheList(KVCacheSimple(), ArraysCache(size: 2))
}

@Suite("Inkling attention")
struct InklingAttentionTests {

    /// Three ways a checkpoint can declare the schedule, and the reference default.
    @Test("Layer typing follows whichever key the checkpoint sets")
    func layerTyping() throws {
        let typed = try inklingConfig(tinyInkling)
        #expect(typed.layerIsSliding(0))
        #expect(typed.layerIsSliding(1) == false)

        let byID = try inklingConfig(
            #"{"num_hidden_layers": 4, "local_layer_ids": [0, 2]}"#)
        #expect(byID.layerIsSliding(0))
        #expect(byID.layerIsSliding(1) == false)
        #expect(byID.layerIsSliding(2))

        // The reference default makes every sixth layer global.
        let byDefault = try inklingConfig(#"{"num_hidden_layers": 12}"#)
        #expect((0 ..< 12).map { byDefault.layerIsSliding($0) }.filter { !$0 }.count == 2)
        #expect(byDefault.layerIsSliding(5) == false)
        #expect(byDefault.layerIsSliding(11) == false)
        #expect(byDefault.layerIsSliding(0))
    }

    /// The stacked projection's width is the sum of four parts, each read from the layer's
    /// own geometry -- a sliding layer here is narrower on every axis.
    @Test("Projection widths follow the layer's own geometry")
    func projectionWidths() throws {
        let config = try inklingConfig(tinyInkling)

        let sliding = InklingAttention(config, layerIndex: 0)
        #expect(sliding.isSliding)
        #expect((sliding.heads, sliding.kvHeads, sliding.headDim) == (2, 1, 4))
        #expect(sliding.slidingWindow == 6)
        // The band is the window on a sliding layer, not rel_extent.
        #expect(sliding.relativeExtent == 6)
        // Log scaling is a global-layer feature.
        #expect(sliding.logScalingFloor == nil)

        let global = InklingAttention(config, layerIndex: 1)
        #expect(global.isSliding == false)
        #expect((global.heads, global.kvHeads, global.headDim) == (4, 2, 8))
        #expect(global.slidingWindow == 0)
        #expect(global.relativeExtent == 5)
        #expect(global.logScalingFloor == 4)

        // Inkling scales by 1 / head_dim, not its inverse square root.
        #expect(global.scale == 1.0 / 8.0)
        #expect(sliding.scale == 1.0 / 4.0)

        let shapes = Dictionary(
            uniqueKeysWithValues: global.parameters().flattened().map { ($0.0, $0.1.shape) })
        // q + k + v + r, stacked.
        #expect(shapes["qkvr_proj.weight"] == [4 * 8 + 2 * 8 + 2 * 8 + 4 * 2, 32])
        #expect(shapes["o_proj.weight"] == [32, 4 * 8])
        #expect(shapes["rel_proj"] == [2, 5])
        #expect(shapes["q_norm.weight"] == [8])
        // The convolutions sit on the key and value widths.
        #expect(shapes["k_sconv.conv.weight"] == [2 * 8, 4, 1])
    }

    /// A prompt pass then a decode step, on both layer kinds: the KV cache and the two
    /// convolution slots all advance.
    @Test("Prompt and decode advance both the KV and convolution state")
    func forward() throws {
        let config = try inklingConfig(tinyInkling)
        for layerIndex in [0, 1] {
            let attention = InklingAttention(config, layerIndex: layerIndex)
            let cache = layerCache()

            let prompt = attention(MLXArray.zeros([1, 5, 32]), cache: cache)
            eval(prompt)
            #expect(prompt.shape == [1, 5, 32])
            #expect(cache[0].offset == 5)

            let step = attention(MLXArray.zeros([1, 1, 32]), cache: cache)
            eval(step)
            #expect(step.shape == [1, 1, 32])
            #expect(cache[0].offset == 6)

            // Both convolutions carry their own window.
            let conv = try #require(cache[1] as? ArraysCache)
            #expect(
                try #require(conv[0]).shape == [1, 3, config.layerIsSliding(layerIndex) ? 4 : 16])
            #expect(conv[1] != nil)
        }
    }

    /// Past its window a sliding layer slices the unreachable keys off rather than masking
    /// them, so the attention narrows instead of growing with the cache.
    @Test("A sliding layer stops growing past its window")
    func slidingSlice() throws {
        let config = try inklingConfig(tinyInkling)
        let attention = InklingAttention(config, layerIndex: 0)
        let cache = layerCache()

        // Twelve positions with a window of six: the last step can only reach six back.
        _ = attention(MLXArray.zeros([1, 12, 32]), cache: cache)
        let step = attention(MLXArray.zeros([1, 1, 32]), cache: cache)
        eval(step)
        #expect(step.shape == [1, 1, 32])
        // The cache itself still holds everything; only the attention window narrowed.
        #expect(cache[0].offset == 13)
    }

}
