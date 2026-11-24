import Foundation
import MLX
import MLXNN
import Testing

@testable import NativSDK

private func inklingConfig(overrides: [String: Any] = [:]) throws -> InklingTextConfiguration {
    try JSONDecoder.json5().decode(
        InklingTextConfiguration.self,
        from: ModelTestConfigurations.configuration("inkling", overrides: overrides))
}

@Suite("Inkling model")
struct InklingModelTests {

    /// A layer holds two convolutions of its own on top of the two inside its attention,
    /// so a layer's cache carries four slots and they must not collide.
    @Test("A layer fills all four convolution slots")
    func layerFillsFourSlots() throws {
        let config = try inklingConfig()
        let layer = InklingDecoderLayer(config, layerIndex: 1)
        let cache = CacheList(KVCacheSimple(), ArraysCache(size: 4))

        let out = layer(MLXArray.zeros([1, 3, 16]), cache: cache)
        eval(out)
        #expect(out.shape == [1, 3, 16])

        let conv = try #require(cache[1] as? ArraysCache)
        for slot in 0 ..< 4 {
            #expect(conv[slot] != nil, "convolution slot \(slot) was never written")
        }
        // The key and value windows are the KV width; the layer's own are the hidden width.
        #expect(try #require(conv[0]).shape == [1, 3, 2 * 4])
        #expect(try #require(conv[2]).shape == [1, 3, 16])
        #expect(cache[0].offset == 3)
    }

    /// Dense and routed layers are picked per index.
    @Test("Feed-forward kind follows the layer schedule")
    func layerSchedule() throws {
        let config = try inklingConfig()
        #expect(config.layerIsDense(0))
        #expect(config.layerIsDense(1) == false)

        let names0 = Set(
            InklingDecoderLayer(config, layerIndex: 0).parameters().flattened().map(\.0))
        #expect(names0.contains("mlp.gate_proj.weight"))
        #expect(names0.contains("mlp.global_scale"))

        let names1 = Set(
            InklingDecoderLayer(config, layerIndex: 1).parameters().flattened().map(\.0))
        #expect(names1.contains("mlp.switch_mlp.gate_proj.weight"))
        #expect(names1.contains("mlp.shared_experts.gate_proj.weight"))
        #expect(names1.contains("mlp.gate_weight"))
    }

    /// The width multiplier divides before the head, so it changes the logits' scale.
    @Test("The width multiplier scales the logits")
    func widthMultiplier() throws {
        let scaled = InklingModel(try inklingConfig())
        let plain = InklingModel(
            try inklingConfig(overrides: ["logits_mup_width_multiplier": 1.0]))
        plain.update(parameters: scaled.parameters())

        let hidden = MLXArray.ones([1, 1, 16])
        let a = scaled.logits(from: hidden)
        let b = plain.logits(from: hidden)
        eval(a, b)
        for (x, y) in zip(
            a.reshaped(-1).asArray(Float.self), b.reshaped(-1).asArray(Float.self))
        {
            #expect(abs(x * 2 - y) < 1e-4)
        }
    }

    /// Loading runs both fusions, so a checkpoint's separate projections and switched
    /// shared experts land in the shapes the model declares.
    @Test("Sanitize applies both load-time fusions")
    func sanitize() throws {
        let model = InklingModel(try inklingConfig())
        var weights: [String: MLXArray] = [
            "model.layers.1.mlp.shared_experts.gate_proj.weight": MLXArray.zeros([2, 8, 16]),
            "model.layers.1.mlp.shared_experts.down_proj.weight": MLXArray.zeros([2, 16, 8]),
        ]
        for (name, rows) in [("q", 16), ("k", 8), ("v", 8), ("r", 8)] {
            weights["model.layers.1.self_attn.\(name)_proj.weight"] = MLXArray.zeros([
                rows, 16,
            ])
        }

        let sanitized = model.sanitize(weights: weights)
        #expect(sanitized["model.layers.1.self_attn.qkvr_proj.weight"]?.shape == [40, 16])
        #expect(
            sanitized["model.layers.1.mlp.shared_experts.gate_proj.weight"]?.shape
                == [16, 16])
        #expect(
            sanitized["model.layers.1.mlp.shared_experts.down_proj.weight"]?.shape
                == [16, 16])
    }
}
